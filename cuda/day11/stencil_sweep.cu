#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__constant__ float c0;
__constant__ float c1;
__constant__ float c2;
__constant__ float c3;
__constant__ float c4;
__constant__ float c5;
__constant__ float c6;

void printMatrix(float *matrix, int width, int height, int depth) {
    printf("\n\n");

    for (int z = 0; z < depth; z++) {
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                printf("%f ", matrix[z * width * height + x * width + y]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n\n");
}

__global__ void stencilKernel(float *in, float *out, int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] +
                                     c1 * in[i * N * N + j * N + (k - 1)] + 
                                     c2 * in[i * N * N + j * N + (k + 1)] + 
                                     c3 * in[i * N * N + (j - 1) * N + k] + 
                                     c4 * in[i * N * N + (j + 1) * N + k] + 
                                     c5 * in[(i - 1) * N * N + j * N + k] + 
                                     c6 * in[(i + 1) * N * N + j * N + k];
    }
}

void stencilSweep(float *in, float *out, int N) {
    int size = N * N * N * sizeof(float);
    float *in_d, *out_d;

    // Initialize constant memory with coefficients. 
    float h_c0 = 0.00f;
    float h_c1 = 0.50f;
    float h_c2 = 0.87f;
    float h_c3 = 1.00f;
    float h_c4 = 0.87f;
    float h_c5 = 0.50f;
    float h_c6 = 0.00f;

    // Part 1: Allocate device memory for input and output.
    cudaError_t err = cudaMalloc((void**)&in_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&out_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // Part 2: Copy input to the device memory.
    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);

    // Part 3: Copy coefficients to the constant memory.
    cudaMemcpyToSymbol(c0, &h_c0, sizeof(float));
    cudaMemcpyToSymbol(c1, &h_c1, sizeof(float));
    cudaMemcpyToSymbol(c2, &h_c2, sizeof(float));
    cudaMemcpyToSymbol(c3, &h_c3, sizeof(float));
    cudaMemcpyToSymbol(c4, &h_c4, sizeof(float));
    cudaMemcpyToSymbol(c5, &h_c5, sizeof(float));
    cudaMemcpyToSymbol(c6, &h_c6, sizeof(float));

    // Part 4: Launch the kernel.
    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid(ceil(N / 8.0), ceil(N / 8.0), ceil(N / 8.0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stencilKernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Part 4: Copy output back to the host memory.
    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    // Part 5: Free the device memory.
    cudaFree(in_d);
    cudaFree(out_d);

    printf("GPU time taken: %f milliseconds\n", milliseconds);
}

int main() {
    int N = 4;

    int size = N * N * N * sizeof(float);
    float *in = (float *)malloc(size);
    float *out = (float *)malloc(size);

    for (int i = 0; i < N * N * N; i++) {
        in[i] = rand() % 2;
    }

    printMatrix(in, N, N, N);

    stencilSweep(in, out, N);

    printMatrix(out, N, N, N);

    free(in);
    free(out);

    return 0;
}