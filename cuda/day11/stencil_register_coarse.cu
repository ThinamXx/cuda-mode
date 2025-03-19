#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IN_TILE_WIDTH 32
#define OUT_TILE_WIDTH 30

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

__global__ void stencilRegisterCoarseKernel(float *in, float *out, int N) {
    // Apply the thread coarsening in the z direction. 

    __shared__ float in_curr_shared[IN_TILE_WIDTH][IN_TILE_WIDTH];

    float inPrev;
    float inCurr;
    float inNext;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int iStart = blockIdx.z * OUT_TILE_WIDTH;
    int j = blockIdx.y * OUT_TILE_WIDTH + ty - 1; // -1 is the order of the stencil. 
    int k = blockIdx.x * OUT_TILE_WIDTH + tx - 1;

    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        in_curr_shared[ty][tx] = inCurr;
    }

    for (int i = iStart; i < iStart + OUT_TILE_WIDTH; ++i) {
        if (i + 1 < N && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (ty >= 1 && ty < IN_TILE_WIDTH - 1 && 
                tx >= 1 && tx < IN_TILE_WIDTH - 1) {
                    out[i * N * N + j * N + k] = c0 * inCurr + 
                                                 c1 * in_curr_shared[ty][tx - 1] + 
                                                 c2 * in_curr_shared[ty][tx + 1] + 
                                                 c3 * in_curr_shared[ty + 1][tx] +
                                                 c4 * in_curr_shared[ty - 1][tx] +
                                                 c5 * inPrev +
                                                 c6 * inNext;
                }
        }
        __syncthreads();

        inPrev = inCurr;
        inCurr = inNext;
        in_curr_shared[ty][tx] = inNext;
    }
}

void stencilRegisterCoarse(float *in, float *out, int N) {
    int size = N * N * N * sizeof(float);

    float *d_in, *d_out;

    // Initialize constant memory with coefficients. 
    float h_c0 = 0.00f;
    float h_c1 = 0.50f;
    float h_c2 = 0.87f;
    float h_c3 = 1.00f;
    float h_c4 = 0.87f;
    float h_c5 = 0.50f;
    float h_c6 = 0.00f;

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

    // 3. Copy coefficients to the constant memory.
    cudaMemcpyToSymbol(c0, &h_c0, sizeof(float));
    cudaMemcpyToSymbol(c1, &h_c1, sizeof(float));
    cudaMemcpyToSymbol(c2, &h_c2, sizeof(float));
    cudaMemcpyToSymbol(c3, &h_c3, sizeof(float));
    cudaMemcpyToSymbol(c4, &h_c4, sizeof(float));
    cudaMemcpyToSymbol(c5, &h_c5, sizeof(float));
    cudaMemcpyToSymbol(c6, &h_c6, sizeof(float));

    // 4. Launch the kernel to perform the stencil computation 
    // with thread coarsening in the z direction. 
    dim3 dimBlock(IN_TILE_WIDTH, IN_TILE_WIDTH, 1);
    dim3 dimGrid(ceil(N / (float)OUT_TILE_WIDTH), ceil(N / (float)OUT_TILE_WIDTH), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stencilRegisterCoarseKernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
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

    stencilRegisterCoarse(in, out, N);

    printMatrix(out, N, N, N);

    free(in);
    free(out);

    return 0;
}