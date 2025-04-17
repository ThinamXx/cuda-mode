#include <cuda_runtime.h>

#define BLOCK_DIM 32
#define COARSE_FACTOR 4

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
        }
    }
}

void cudaSoftmaxReduce(float *input, float *output, int seq_len) {
    const dim3 blockDim(BLOCK_DIM, 1, 1);
    const dim3 gridDim(ceil(seq_len / float(BLOCK_DIM * COARSE_FACTOR * 2)), seq_len, 1);
    const dim3 softmaxGrid(ceil(seq_len / (float)BLOCK_DIM), seq_len, 1);

    float *d_max, *d_sum;
    cudaMalloc((void**)&d_max, seq_len * sizeof(float));
    cudaMalloc((void**)&d_sum, seq_len * sizeof(float));

    max_kernel<<<gridDim, blockDim>>>(input, d_max, seq_len);
    exp_sum_kernel<<<gridDim, blockDim>>>(input, d_max, d_sum, seq_len);
    softmax_kernel<<<softmaxGrid, blockDim>>>(input, output, d_max, d_sum, seq_len);
    cudaDeviceSynchronize();

    cudaFree(d_max);
    cudaFree(d_sum);
}
