#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 4 // Since, we have N = 8, each thread will process 2 elements.

__global__ void sumReductionShared(float *input, float *output, int N) {
    __shared__ float shared_data[BLOCK_DIM];

    unsigned int tid = threadIdx.x;

    shared_data[tid] = input[tid] + input[tid + BLOCK_DIM];
    __syncthreads();

    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = shared_data[0];
    }
}

void sumReduction(float *input, float *output_optimized, int N) {
    int size = N * sizeof(float);

    float *d_input_optimized, *d_output_optimized;

    // 1. Allocate device memory for the input and output arrays.
    cudaError_t err = cudaMalloc((void**)&d_input_optimized, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output_optimized, sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input array to the device memory.
    cudaMemcpy(d_input_optimized, input, size, cudaMemcpyHostToDevice);

    // 3. Launch the kernel. 
    dim3 block(BLOCK_DIM, 1, 1);
    dim3 grid(1, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sumReductionShared<<<grid, block>>>(d_input_optimized, d_output_optimized, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 4. Copy the output array to the host memory.
    cudaMemcpy(output_optimized, d_output_optimized, sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_input_optimized);
    cudaFree(d_output_optimized);
    printf("GPU time taken: %f milliseconds\n", milliseconds);
}

int main() {
    int N = 8;
    float input[N] = {4.0, 7.0, 2.0, 3.0, 8.0, 5.0, 9.0, 6.0};
    
    float *output_optimized = (float *)malloc(sizeof(float));

    sumReduction(input, output_optimized, N);

    printf("Sum of the input optimized: %f\n", output_optimized[0]);

    free(output_optimized);
    return 0;
}