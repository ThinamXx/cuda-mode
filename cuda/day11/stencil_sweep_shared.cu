#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define c0 0.00f
#define c1 0.50f
#define c2 0.87f
#define c3 1.00f
#define c4 0.87f
#define c5 0.50f
#define c6 0.00f

#define IN_TILE_WIDTH 8
#define OUT_TILE_WIDTH 6

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

__global__ void stencilSweepSharedKernel(float *in, float *out, int N) {
    __shared__ float input_tile[IN_TILE_WIDTH][IN_TILE_WIDTH][IN_TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int i = blockIdx.z * OUT_TILE_WIDTH + tz - 1; // 1 is the order of the stencil. 
    int j = blockIdx.y * OUT_TILE_WIDTH + ty - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH + tx - 1;

    if (i >= 0  && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        input_tile[tz][ty][tx] = in[i * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (tz >= 1 && tz < IN_TILE_WIDTH - 1 && 
            ty >= 1 && ty < IN_TILE_WIDTH - 1 && 
            tx >= 1 && tx < IN_TILE_WIDTH - 1) {
                out[i * N * N + j * N + k] = c0 * input_tile[tz][ty][tx] + 
                                             c1 * input_tile[tz][ty][tx - 1] + 
                                             c2 * input_tile[tz][ty][tx + 1] + 
                                             c3 * input_tile[tz][ty - 1][tx] + 
                                             c4 * input_tile[tz][ty + 1][tx] + 
                                             c5 * input_tile[tz - 1][ty][tx] + 
                                             c6 * input_tile[tz + 1][ty][tx];
            }
    }
}

void stencilSweepShared(float *in, float *out, int N) {
    int size = N * N * N * sizeof(float);

    float *d_in, *d_out;

    // 1. Allocate device memory for the input and output arrays.
    cudaError_t err = cudaMalloc((void**)&d_in, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_out, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input array to the device memory.
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    // 3. Launch the kernel to perform the stencil computation.
    dim3 dimBlock(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);
    dim3 dimGrid(ceil(N / (float)IN_TILE_WIDTH), ceil(N / (float)IN_TILE_WIDTH), ceil(N / (float)IN_TILE_WIDTH));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stencilSweepSharedKernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 4. Copy the output array to the host memory.
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_in);
    cudaFree(d_out);

    printf("GPU time taken: %f milliseconds\n", milliseconds);
}

int main() {
    int N = 16;

    int size = N * N * N * sizeof(float);
    float *in = (float *)malloc(size);
    float *out = (float *)malloc(size);

    for (int i = 0; i < N * N * N; i++) {
        in[i] = rand() % 2;
    }

    printMatrix(in, N, N, N);

    stencilSweepShared(in, out, N);

    printMatrix(out, N, N, N);

    free(in);
    free(out);

    return 0;
}