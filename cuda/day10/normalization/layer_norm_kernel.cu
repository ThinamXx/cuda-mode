#include <cuda_runtime.h>

__global__ void layerNorm_kernel(
    float *x,
    float *y,
    float *gamma,
    float *beta,
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;
    
    __shared__ float s_mean;
    __shared__ float s_variance;

    if (threadIdx.x == 0) {
        // 1. Calculate the mean across the embedding dimension.
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            int idx = b * seq_len * embed_dim + s * embed_dim + i;
            sum += x[idx];
        }
        s_mean = sum / embed_dim;

        // 2. Calculate the variance across the embedding dimension.
        float sum_variance = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            int idx = b * seq_len * embed_dim + s * embed_dim + i;
            float diff = x[idx] - s_mean;
            sum_variance += diff * diff;
        }
        s_variance = sum_variance / embed_dim;
    }

    // Ensure the mean and variance are available to all threads.
    __syncthreads();

    if (d < embed_dim) {
        // 3. Apply the normalization formula.
        int index = b * seq_len * embed_dim + s * embed_dim + d;
        float normalized = (x[index] - s_mean) / sqrtf(s_variance + eps);
        y[index] = gamma[d] * normalized + beta[d];
    }
}

void cudaLayerNorm(
    float *x,
    float *y,
    float *gamma,
    float *beta,
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps
) {
    const dim3 blockDim(embed_dim, 1, 1);
    const dim3 gridDim(batch_size, seq_len, 1);

    layerNorm_kernel<<<gridDim, blockDim>>>(x, y, gamma, beta, batch_size, seq_len, embed_dim, eps);
    cudaDeviceSynchronize();
}