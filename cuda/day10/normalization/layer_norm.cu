#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

void layerNorm_CPU(
    float *x,
    float *y, 
    int batch_size, 
    int seq_len,
    int embed_dim, 
    float eps,
    float *gamma,
    float *beta
) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // 1. Calculate mean across the embedding dimension.
            float mean = 0.0f;
            for (int d = 0; d < embed_dim; d++) {
                int index = b * seq_len * embed_dim + s * embed_dim + d;
                mean += x[index];
            }
            mean /= embed_dim;

            // 2. Calculate variance across the embedding dimension.
            // The formula for variance is:
            // variance = sum((x - mean)^2) / embed_dim
            float variance = 0.0f;
            for (int d = 0; d < embed_dim; d++) {
                int index = b * seq_len * embed_dim + s * embed_dim + d;
                float diff = x[index] - mean; 
                variance += diff * diff;
            }
            variance /= embed_dim;

            // 3. Apply normalization formula.
            // The formula for normalization is:
            // y = (x - mean) / sqrt(variance + eps) * gamma + beta
            float sum = 0.0f;
            float sum_squared = 0.0f;
            for (int d = 0; d < embed_dim; d++) {
                int index = b * seq_len * embed_dim + s * embed_dim + d;
                float normalized = (x[index] - mean) / sqrtf(variance + eps);
                y[index] = gamma[d] * normalized + beta[d];

                sum += normalized;
                sum_squared += normalized * normalized;
            }
            printf("\n\n");
            printf("Mean check: %f, variance check: %f", sum / embed_dim, sum_squared / embed_dim);
            printf("\n\n");
        }
    }
}

__global__ void layerNorm_kernel(
    float *x,
    float *y,
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps,
    float *gamma,
    float *beta
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

void layerNorm(
    float *x,
    float *y,
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps,
    float *gamma,
    float *beta
) {
    int size = batch_size * seq_len * embed_dim * sizeof(float);

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
    err = cudaMalloc((void**)&d_gamma, embed_dim * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_beta, embed_dim * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy input to device memory.
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, embed_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Call the kernel to launch the grid of threads.
    dim3 blockDim(embed_dim, 1, 1);
    dim3 gridDim(batch_size, seq_len, 1);
    layerNorm_kernel<<<gridDim, blockDim>>>(d_x, d_y, batch_size, seq_len, embed_dim, eps, d_gamma, d_beta);
    cudaDeviceSynchronize();

    // 4. Copy output from device memory.
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // 5. Free device memory.
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

    float *x = (float *)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *y = (float *)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *cpu_y = (float *)malloc(batch_size * seq_len * embed_dim * sizeof(float));

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

    layerNorm(x, y, batch_size, seq_len, embed_dim, eps, gamma, beta);
    layerNorm_CPU(x, cpu_y, batch_size, seq_len, embed_dim, eps, gamma, beta);

    printf("\nLayerNorm CPU Output:\n");
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        printf("%f ", cpu_y[i]);
    }
    printf("\n\n");

    printf("\nLayerNorm GPU Output:\n");
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

    return 0;
}