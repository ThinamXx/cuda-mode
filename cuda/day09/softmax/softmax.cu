#include <cuda_runtime.h>

__global__ void softmax_kernel(float *x, float *y, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < seq_len) {
        // 1. Compute max and sum of exp(x[idx][i] - max_val).
        float max_val = -INFINITY;
        float sum_val = 0.0f;

        for (int i = 0; i < seq_len; i++) {
            float cur_val = x[idx * seq_len + i];
            float max_cur = fmaxf(max_val, cur_val);
            float sum_cur = sum_val * expf(max_val - max_cur) + expf(cur_val - max_cur);
            max_val = max_cur;
            sum_val = sum_cur;
        }
        
        // 2. Compute softmax.
        for (int i = 0; i < seq_len; i++) {
            y[idx * seq_len + i] = expf(x[idx * seq_len + i] - max_val) / sum_val;
        }
    }
}

void cudaSoftmax(float *input, float *output, int seq_len) {
    const dim3 blockDim(32, 1, 1);
    const dim3 gridDim(ceil(seq_len / 32.0), 1, 1);

    softmax_kernel<<<gridDim, blockDim>>>(input, output, seq_len);
    cudaDeviceSynchronize();
}
