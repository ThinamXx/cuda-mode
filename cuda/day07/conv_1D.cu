#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

void print_1D_array(float *array, int n) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void conv_1D_CPU(float *input_1D, float *filter_1D, float *output_1D, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;

        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            int index = i + j;
            int value = (index >= 0 && index < n) ? input_1D[index] : 0.0f;
            sum += value * filter_1D[j + FILTER_RADIUS];
        }
        output_1D[i] = sum;
    }
}

__global__ void conv_1D_kernel(float *input_1D, float *filter_1D, float *output_1D, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float sum = 0.0f;
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            int index = tid + i;
            // If the index is out of bounds, set the value to 0.
            int value = (index >= 0 && index < n) ? input_1D[index] : 0.0f;
            sum += value * filter_1D[i + FILTER_RADIUS];
        }
        output_1D[tid] = sum;
    }
}

void conv_1D(float *input_1D, float *filter_1D, float *output_1D, int n) {
    int size = n * sizeof(float);

    float *d_input_1D, *d_filter_1D, *d_output_1D;

    // 1. Allocate device memory for input, filter, and output
    // copy input and filter to device memory. 
    cudaError_t err = cudaMalloc((void**)&d_input_1D, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_filter_1D, FILTER_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output_1D, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(d_input_1D, input_1D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_1D, filter_1D, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Call the kernel to launch the grid of threads 
    // to perform the convolution.
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(ceil(n / (float)dimBlock.x), 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv_1D_kernel<<<dimGrid, dimBlock>>>(d_input_1D, d_filter_1D, d_output_1D, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 3. Copy the output back to the host
    // free the device memory. 
    cudaMemcpy(output_1D, d_output_1D, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input_1D);
    cudaFree(d_filter_1D);
    cudaFree(d_output_1D);

    printf("\nGPU Time: %0.2f ms\n", milliseconds);
}

int main() {
    int n = 7;
    float input_1D[n] = {1, 2, 3, 4, 5, 6, 7};
    float filter_1D[FILTER_SIZE] = {1, 2, 3, 4, 5};
    float output_1D[n], output_1D_CPU[n];

    print_1D_array(input_1D, n);
    print_1D_array(filter_1D, FILTER_SIZE);
 
    conv_1D(input_1D, filter_1D, output_1D, n);
    conv_1D_CPU(input_1D, filter_1D, output_1D_CPU, n);

    // Compare the results:
    for (int i = 0; i < n; i++) {
        if (abs(output_1D[i] - output_1D_CPU[i]) > 1e-6) {
            printf("\nError at index %d: %f (GPU) != %f (CPU)\n", i, output_1D[i], output_1D_CPU[i]);
        } else {
            printf("\nOutput at index %d: %f (GPU) == %f (CPU)\n", i, output_1D[i], output_1D_CPU[i]);
        }
    }

    return 0;
}