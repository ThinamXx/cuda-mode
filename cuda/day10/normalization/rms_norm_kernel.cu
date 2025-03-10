#include <cuda_runtime.h>

__global__ void rmsNorm_kernel(
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

    // I was thinking of using shared memory for gamma and beta, but 
    // sometimes the size of gamma and beta is too large to fit into 
    // the shared memory so only keeping rms_score in the shared memory.
    __shared__ float s_rms_score;

    if (threadIdx.x == 0) {
        // 1. Calculate the rms score across the embedding dimension.
        float rms_score = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            int idx = b * seq_len * embed_dim + s * embed_dim + i;
            rms_score += x[idx] * x[idx];
        }
        s_rms_score = sqrtf((rms_score / embed_dim) + eps);
    }

    // Ensure the rms_score is available to all threads.
    __syncthreads();

    if (d < embed_dim) {
        int index = b * seq_len * embed_dim + s * embed_dim + d;
        y[index] = gamma[d] * (x[index] / s_rms_score) + beta[d];
    }   
}

void cudaRMSNorm(
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

    rmsNorm_kernel<<<gridDim, blockDim>>>(x, y, gamma, beta, batch_size, seq_len, embed_dim, eps);
    cudaDeviceSynchronize();
}
