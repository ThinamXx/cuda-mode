#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

void printMatrix(float *matrix, int width, int height) {
    printf("\n\n");

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

__host__ int get_optimal_tile_size(cudaDeviceProp device_prop) {
    // Verify this GPU has 32-thread warps.
    if (device_prop.warpSize != 32) {
        printf("Warning: Unexpected warp size %d detected. Optimal tile size may differ.\n", 
               device_prop.warpSize);
    }
    
    // Check if the device can support a 32×32 tile.
    size_t shared_mem_needed = 2 * 32 * 32 * sizeof(float); // For two 32×32 tiles
    if (shared_mem_needed > device_prop.sharedMemPerBlock) {
        printf("Warning: Device has insufficient shared memory for 32×32 tiles.\n");
        return 16;
    }
    
    // Check thread count constraints.
    if (32 * 32 > device_prop.maxThreadsPerBlock) {
        printf("Warning: Device cannot support 1024 threads per block.\n");
        return 16;
    }
    
    printf("Using optimal tile size: 32\n");
    return 32;
}

__global__ void selfAttentionScore_kernel(
    float *query,
    float *key,
    float *attention_scores,
    int seq_len,
    int embed_dim,
    int tile_width,
    unsigned query_tile_offset
) {
    extern __shared__ float shared_mem[];

    float *query_tile = (float *)(shared_mem);
    float *key_tile = (float *)(shared_mem + query_tile_offset / sizeof(float));

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    float sum_val = 0.0f;
    for (int ph = 0; ph < ceil(embed_dim / (float)tile_width); ++ph) {
        if ((row < seq_len) && (ph * tile_width + tx < embed_dim)) {
            query_tile[ty * tile_width + tx] = query[row * embed_dim + ph * tile_width + tx];
        } else {
            query_tile[ty * tile_width + tx] = 0.0f;
        }

        // We need to understand that we should be using the corner turning algorithm
        // while loading the key matrix to the shared memory because the query and key 
        // matrices are of shape (seq_len, embed_dim) and (seq_len, embed_dim) respectively.
        // And we need to transpose the key matrix to make the dot product operation. 
        if (((ph * tile_width + ty) < seq_len) && (col < embed_dim)) {
            key_tile[tx * tile_width + ty] = key[(ph * tile_width + ty) * embed_dim + col];
        } else {
            key_tile[tx * tile_width + ty] = 0.0f;
        }

        __syncthreads();
        
        for (int k = 0; k < tile_width; ++k) {
            sum_val += query_tile[ty * tile_width + k] * key_tile[k * tile_width + tx];
        }

        __syncthreads();
    }

    if ((row < seq_len) && (col < seq_len)) {
        attention_scores[row * seq_len + col] = sum_val;
    }
}

__global__ void selfAttentionSoftmax_kernel(
    float *attention_scores,
    float *attention_scores_softmax,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < seq_len) {
        // 1. Compute max and sum of exp(attention_scores[idx][i] - max_val).
        float max_val = -INFINITY;
        float sum_val = 0.0f;

        for (int i = 0; i < seq_len; i++) {
            float cur_val = attention_scores[idx * seq_len + i];
            float max_cur = fmaxf(max_val, cur_val);
            float sum_cur = sum_val * expf(max_val - max_cur) + expf(cur_val - max_cur);
            max_val = max_cur;
            sum_val = sum_cur;
        }
        
        // 2. Compute softmax.
        for (int i = 0; i < seq_len; i++) {
            attention_scores_softmax[idx * seq_len + i] = expf(attention_scores[idx * seq_len + i] - max_val) / sum_val;
        }
    }
}

__global__ void selfAttentionOutput_kernel(
    float *attention_scores, 
    float *value,
    float *attention_output,
    int seq_len,
    int embed_dim,
    int tile_width,
    unsigned attention_scores_tile_offset
) {
    extern __shared__ float shared_mem[];

    float *attention_scores_tile = (float *)(shared_mem);
    float *value_tile = (float *)(shared_mem + attention_scores_tile_offset / sizeof(float));

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    float sum_val = 0.0f;
    for (int ph = 0; ph < ceil(seq_len / (float)tile_width); ++ph) {
        if ((row < seq_len) && (ph * tile_width + tx < seq_len)) {
            attention_scores_tile[ty * tile_width + tx] = attention_scores[row * seq_len + ph * tile_width + tx];
        } else {
            attention_scores_tile[ty * tile_width + tx] = 0.0f;
        }

        if (((ph * tile_width + ty) < seq_len) && (col < embed_dim)) {
            value_tile[ty * tile_width + tx] = value[(ph * tile_width + ty) * embed_dim + col];
        } else {
            value_tile[ty * tile_width + tx] = 0.0f;
        }

        __syncthreads();
        
        for (int k = 0; k < tile_width; ++k) {
            sum_val += attention_scores_tile[ty * tile_width + k] * value_tile[k * tile_width + tx];
        }

        __syncthreads();
    }

    if ((row < seq_len) && (col < embed_dim)) {
        attention_output[row * embed_dim + col] = sum_val;
    }
}
   
void selfAttention(
    float *query,
    float *key,
    float *value,
    float *attention_scores,
    float *attention_output,
    int seq_len,
    int embed_dim
) {
    int size = seq_len * embed_dim * sizeof(float);

    float *d_query, *d_key, *d_value, *d_attention_scores, *d_attention_output;

    // Determine the appropriate tile size. 
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0); // 0 means the first GPU.
    int tile_width = get_optimal_tile_size(device_prop); // Usually 32 for all NVIDIA GPUs.

    // 1. Allocate device memory for input, output.
    cudaError_t err = cudaMalloc((void**)&d_query, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_key, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_value, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_attention_scores, seq_len * seq_len * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_attention_output, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    
    // 2. Copy the input to the device.
    cudaMemcpy(d_query, query, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, size, cudaMemcpyHostToDevice);

    // 3. Call the self attention kernel in 3 steps.
    // Step 1: Compute the attention scores query * key^T.
    dim3 dimBlock(tile_width, tile_width, 1);
    dim3 dimGrid_seqlen(ceil(seq_len / (float)tile_width), ceil(seq_len / (float)tile_width), 1);
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);

    selfAttentionScore_kernel<<<dimGrid_seqlen, dimBlock, shared_mem_size>>>(d_query, d_key, d_attention_scores, seq_len, embed_dim, tile_width, shared_mem_size/2);
    cudaDeviceSynchronize();

    printf("Attention Scores before softmax:\n");
    float *temp_attention_scores = (float *)malloc(seq_len * seq_len * sizeof(float));
    cudaMemcpy(temp_attention_scores, d_attention_scores, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(temp_attention_scores, seq_len, seq_len);

    // Step 2: Apply the softmax on the attention scores.
    dim3 dimBlock_softmax(256, 1, 1);
    dim3 dimGrid_softmax(ceil(seq_len / (float)256), 1, 1);

    selfAttentionSoftmax_kernel<<<dimGrid_softmax, dimBlock_softmax>>>(d_attention_scores, d_attention_scores, seq_len);
    cudaDeviceSynchronize();

    // Step 3: Compute the attention output.
    dim3 dimGrid_embeddim(ceil(seq_len / (float)tile_width), ceil(embed_dim / (float)tile_width), 1);

    selfAttentionOutput_kernel<<<dimGrid_embeddim, dimBlock, shared_mem_size>>>(d_attention_scores, d_value, d_attention_output, seq_len, embed_dim, tile_width, shared_mem_size/2);
    cudaDeviceSynchronize();

    // 4. Copy the output from the device to the host.
    cudaMemcpy(attention_scores, d_attention_scores, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(attention_output, d_attention_output, size, cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_attention_scores);
    cudaFree(d_attention_output);

    free(temp_attention_scores);
}

int main() {
    int seq_len = 2;
    int embed_dim = 2;

    int size = seq_len * embed_dim * sizeof(float);

    float *query = (float *)malloc(size);
    float *key = (float *)malloc(size);
    float *value = (float *)malloc(size);

    float *attention_scores = (float *)malloc(seq_len * seq_len * sizeof(float));
    float *attention_output = (float *)malloc(size);

    // Initialize the query, key and value arrays with random values.
    for (int i = 0; i < seq_len * embed_dim; i++) {
        query[i] = rand() % 2;
        key[i] = rand() % 2;
        value[i] = rand() % 2; 
    }

    printf("Query:\n");
    printMatrix(query, embed_dim, seq_len);
    printf("Key:\n");
    printMatrix(key, embed_dim, seq_len);
    printf("Value:\n");
    printMatrix(value, embed_dim, seq_len);

    // Call the self attention kernel.
    selfAttention(query, key, value, attention_scores, attention_output, seq_len, embed_dim);

    printf("Attention Scores:\n");
    printMatrix(attention_scores, seq_len, seq_len);
    printf("Attention Output:\n");
    printMatrix(attention_output, embed_dim, seq_len);

    // Free the allocated memory.
    free(query);
    free(key);
    free(value);
    free(attention_scores);
    free(attention_output);

    return 0;
}