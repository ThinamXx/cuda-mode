#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumReductionKernel(float *input, float *output, int N) {
    unsigned int tid = threadIdx.x;

    for (unsigned int stride = 1; stride < N; stride *= 2) {
        int index = 2 * stride * tid; // tid is the offset. 
        if (index + stride < N) {
            input[index] += input[index + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void sumReductionKernelOptimized(float *input, float *output, int N) {
    unsigned int tid = threadIdx.x;

    for (unsigned int stride = N / float(2); stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[tid] += input[tid + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

void sumReduction(float *input, float *output, float *output_optimized, int N) {
    int size = N * sizeof(float);

    float *d_input, *d_input_optimized, *d_output, *d_output_optimized;

    // 1. Allocate device memory for the input and output arrays.
    cudaError_t err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_input_optimized, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output, sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output_optimized, sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input array to the device memory.
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_optimized, input, size, cudaMemcpyHostToDevice);

    // 3. Launch the kernel. 
    dim3 block(N / 2, 1, 1);
    dim3 grid(1, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sumReductionKernel<<<grid, block>>>(d_input, d_output, N);
    sumReductionKernelOptimized<<<grid, block>>>(d_input_optimized, d_output_optimized, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 4. Copy the output array to the host memory.
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_optimized, d_output_optimized, sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_optimized);
    printf("GPU time taken: %f milliseconds\n", milliseconds);
}

int main() {
    // Since, reduction requires collaboration between threads, 
    // we will use 1 block of threads. The max number of threads 
    // in a block is 1024, means that we can process 2*1024 elements. 
    int N = 8;
    float input[N] = {4.0, 7.0, 2.0, 3.0, 8.0, 5.0, 9.0, 6.0};
    
    float *output = (float *)malloc(sizeof(float));
    float *output_optimized = (float *)malloc(sizeof(float));

    sumReduction(input, output, output_optimized, N);

    printf("Sum of the input: %f\n", output[0]);
    printf("Sum of the input optimized: %f\n", output_optimized[0]);

    free(output);
    free(output_optimized);
    return 0;
}