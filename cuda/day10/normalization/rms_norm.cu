#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

// You can take the reference for RMSNorm from here: 
// https://github.com/ThinamXx/Meta-llama/blob/main/llama/llama.py#L80C1-L89C63

void rmsNorm_CPU(
    float *x,
    float *y,
    float *gamma,
    float *beta,
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps
) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // 1. Calculate the rms score across the embedding dimension.
            float rms_score = 0.0f;
            for (int d = 0; d < embed_dim; d++) {
                int index = b * seq_len * embed_dim + s * embed_dim + d;
                rms_score += x[index] * x[index];
            }
            rms_score /= embed_dim;
            rms_score = sqrtf(rms_score + eps);

            // 2. Apply the gamma and beta to the rms score.
            for (int d = 0; d < embed_dim; d++) {
                int index = b * seq_len * embed_dim + s * embed_dim + d;
                y[index] = gamma[d] * (x[index] / rms_score) + beta[d];
            }
        }
    }
}

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

void rmsNorm(
    float *x,
    float *y,
    float *gamma,
    float *beta,
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps
) {
    int size = batch_size * seq_len * embed_dim * sizeof(float);
    int embed_dim_size = embed_dim * sizeof(float);

    float *d_x, *d_y, *d_gamma, *d_beta;

    // 1. Allocate device memory for input, output, gamma, beta.
    cudaError_t err = cudaMalloc((void**)&d_x, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_y, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_gamma, embed_dim_size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_beta, embed_dim_size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input, gamma, beta to the device.
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, embed_dim_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, embed_dim_size, cudaMemcpyHostToDevice);

    // 3. Call the kernel to launch the grid of threads.
    dim3 blockDim(embed_dim, 1, 1);
    dim3 gridDim(batch_size, seq_len, 1);
    rmsNorm_kernel<<<gridDim, blockDim>>>(d_x, d_y, d_gamma, d_beta, batch_size, seq_len, embed_dim, eps);
    cudaDeviceSynchronize();

    // 4. Copy the output from the device to the host.
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

int main() {
    int batch_size = 2;
    int seq_len = 3;
    int embed_dim = 4;
    float eps = 1e-5;

    int size = batch_size * seq_len * embed_dim * sizeof(float);

    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);
    float *cpu_y = (float *)malloc(size);

    // Since, we are normalizing along the embedding or last dimension, 
    // the gemma and beta will be of size embed_dim. 
    float *gamma = (float *)malloc(embed_dim * sizeof(float));
    float *beta = (float *)malloc(embed_dim * sizeof(float));

    // Initialize the input array with random values.
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        x[i] = rand() % 2;
    }

    // Initialize the gamma and beta arrays where the gamma is the 
    // multiplicative factor and beta is the additive factor.
    for (int i = 0; i < embed_dim; i++) {
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }

    rmsNorm(x, y, gamma, beta, batch_size, seq_len, embed_dim, eps);
    rmsNorm_CPU(x, cpu_y, gamma, beta, batch_size, seq_len, embed_dim, eps);

    printf("\nRMSNorm CPU Output:\n");
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        printf("%f ", cpu_y[i]);
    }
    printf("\n\n");

    printf("\nRMSNorm GPU Output:\n");
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        printf("%f ", y[i]);
    }
    printf("\n\n");

    // Check if the output of the GPU and CPU are the same.
    bool is_same = true;
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        if (fabs(cpu_y[i] - y[i]) > 1e-5) {
            is_same = false;
            break;
        }
    }

    if (is_same) {
        printf("The output of the GPU and CPU are the same.\n");
    } else {
        printf("The output of the GPU and CPU are different.\n");
    }

    free(x);
    free(y);
    free(cpu_y);
    free(gamma);
    free(beta);

    return 0;
}