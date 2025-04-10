#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 4 
#define COARSE_FACTOR 2

__global__ void sumReductionCoarse(float *input, float *output) {
    __shared__ float shared_data[BLOCK_DIM];

    unsigned int segment = COARSE_FACTOR * 2 * BLOCK_DIM * blockIdx.x;

    unsigned int segment_tid = segment + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float sum = input[segment_tid];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) { // each thread will process COARSE_FACTOR * 2 elements.
        sum += input[segment_tid + tile * BLOCK_DIM];
    }

    shared_data[tid] = sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
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
    dim3 grid(ceil(N / float(BLOCK_DIM * COARSE_FACTOR * 2)), 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sumReductionCoarse<<<grid, block>>>(d_input_optimized, d_output_optimized);
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