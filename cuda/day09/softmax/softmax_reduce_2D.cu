#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_DIM 32
#define COARSE_FACTOR 4

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

void cpuSoftmax(float *input, float *output, int seq_len) {
    for (int row = 0; row < seq_len; row++) {

        // 1. Find max value.
        float max_val = input[row * seq_len];
        for (int col = 0; col < seq_len; col++) {
            if (input[row * seq_len + col] > max_val) {
                max_val = input[row * seq_len + col];
            }
        }

        // 2. Compute sum of exp(x[row][col] - max_val).
        float sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            sum += expf(input[row * seq_len + col] - max_val);
        }

        // 3. Compute softmax.
        for (int col = 0; col < seq_len; col++) {
            output[row * seq_len + col] = expf(input[row * seq_len + col] - max_val) / sum;
        }
    }
}

__global__ void max_kernel(float *input, float *output, int seq_len) {
    __shared__ float shared_data[BLOCK_DIM];

    int row = blockIdx.y;
    int base_idx = row * seq_len;
    
    unsigned int segment = COARSE_FACTOR * 2 * BLOCK_DIM * blockIdx.x;
    unsigned int segment_tid = segment + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    if (segment_tid < seq_len) {
        max_val = input[base_idx + segment_tid];

        for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
            unsigned int idx = segment_tid + tile * BLOCK_DIM;
            if (idx < seq_len) {
                max_val = fmaxf(max_val, input[base_idx + idx]);
            }
        }
    }

    shared_data[tid] = max_val;
    __syncthreads();

    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)&output[row], __float_as_int(shared_data[0]));
    }
}

__global__ void exp_sum_kernel(float *input, float *max_val, float *output, int seq_len) {
    __shared__ float shared_data[BLOCK_DIM];

    int row = blockIdx.y;
    int base_idx = row * seq_len;
    float row_max = max_val[row];
    
    unsigned int segment = COARSE_FACTOR * 2 * BLOCK_DIM * blockIdx.x;
    unsigned int segment_tid = segment + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float exp_sum_val = 0.0f;
    if (segment_tid < seq_len) {
        exp_sum_val = __expf(input[base_idx + segment_tid] - row_max);

        for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
            unsigned int idx = segment_tid + tile * BLOCK_DIM;
            if (idx < seq_len) {
                exp_sum_val += __expf(input[base_idx + idx] - row_max);
            }
        }
    }

    shared_data[tid] = exp_sum_val;
    __syncthreads();

    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&output[row], shared_data[0]);
    }
}

__global__ void softmax_kernel(float *input, float *output, float *max_val, float *sum, int seq_len) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < seq_len) {
        int idx = row * seq_len + col;
        float row_max = max_val[row];
        float row_sum = sum[row];
        
        if (row_sum > 0.0f) {
            output[idx] = __expf(input[idx] - row_max) / row_sum;
        } else {
            output[idx] = (col == 0) ? 1.0f : 0.0f;
        }
    }
}

void softmax(float *input, float *output, int seq_len) {
    float *d_input, *d_output;
    float *d_max, *d_sum;

    // 1. Allocate device memory.
    cudaMalloc((void**)&d_input, seq_len * seq_len * sizeof(float));
    cudaMalloc((void**)&d_output, seq_len * seq_len * sizeof(float));
    cudaMalloc((void**)&d_max, seq_len * sizeof(float));
    cudaMalloc((void**)&d_sum, seq_len * sizeof(float));

    // 2. Copy input to device memory.
    cudaMemcpy(d_input, input, seq_len * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Initialize the max and sum values on device.
    cudaMemset(d_max, 0, seq_len * sizeof(float));
    cudaMemset(d_sum, 0, seq_len * sizeof(float));

    // 4. Launch the kernel to find the max value.
    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(ceil(seq_len / float(BLOCK_DIM * COARSE_FACTOR * 2)), seq_len, 1);

    max_kernel<<<gridDim, blockDim>>>(d_input, d_max, seq_len);
    
    float *h_max = (float *)malloc(seq_len * sizeof(float));
    cudaMemcpy(h_max, d_max, seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Compute the sum of exponentials. 
    cudaMemset(d_sum, 0, seq_len * sizeof(float));

    exp_sum_kernel<<<gridDim, blockDim>>>(d_input, d_max, d_sum, seq_len);
    
    float *h_sum = (float *)malloc(seq_len * sizeof(float));
    cudaMemcpy(h_sum, d_sum, seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Compute the softmax values.
    dim3 softmaxBlock(BLOCK_DIM, 1, 1);
    dim3 softmaxGrid(ceil(seq_len / (float)BLOCK_DIM), seq_len, 1);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    softmax_kernel<<<softmaxGrid, softmaxBlock>>>(d_input, d_output, d_max, d_sum, seq_len);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Softmax kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(output, d_output, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max);
    cudaFree(d_sum);
    free(h_max);
    free(h_sum);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int seq_len = 10;

    float *input = (float *)malloc(seq_len * seq_len * sizeof(float));
    float *output = (float *)malloc(seq_len * seq_len * sizeof(float));
    float *cpu_output = (float *)malloc(seq_len * seq_len * sizeof(float));

    for (int i = 0; i < seq_len * seq_len; i++) {
        input[i] = rand() % 2;
    }

    printMatrix(input, seq_len, seq_len);

    softmax(input, output, seq_len);
    cpuSoftmax(input, cpu_output, seq_len);

    printMatrix(output, seq_len, seq_len);
    printMatrix(cpu_output, seq_len, seq_len);

    // Check if the results are the same.
    bool same = true;
    for (int i = 0; i < seq_len * seq_len; i++) {
        if (abs(output[i] - cpu_output[i]) > 1e-6) {
            printf("Error at index %d: %f != %f\n", i, output[i], cpu_output[i]);
            same = false;
            break;
        }
    }
    if (same) {
        printf("Results are the same.\n");
    } else {
        printf("Results are not the same.\n");
    }

    free(input);
    free(output);
    free(cpu_output);
    return 0;
}

