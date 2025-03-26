#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BUCKETS 7

__global__ void aggregationKernel(char *data, unsigned int *histogram, unsigned int N) {
    __shared__ unsigned int hist_s[NUM_BUCKETS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
        hist_s[bin] = 0u;
    }
    __syncthreads();

    unsigned int accumulator = 0u;
    int prevBin = -1;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < N; i += blockDim.x * gridDim.x) {
        int alpha_index = data[i] - 'a';
        if (alpha_index >= 0 && alpha_index < 26) {
            int bin = alpha_index / 4;
            if (bin == prevBin) {
                ++accumulator;
            } else {
                if (accumulator > 0) {
                    atomicAdd(&(hist_s[prevBin]), accumulator);
                }
                accumulator = 1;
                prevBin = bin;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&(hist_s[prevBin]), accumulator);
    }

    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
        unsigned int binValue = hist_s[bin];
        if (binValue > 0) {
            atomicAdd(&(histogram[bin]), binValue);
        }
    }
}

void histogramCuda(char *data, unsigned int *histogram, unsigned int N) {
    int size = N * sizeof(char);
    int size_histogram = NUM_BUCKETS * sizeof(unsigned int);

    char *d_data;
    unsigned int *d_histogram;

    // 1. Allocate device memory for the input and output arrays.
    cudaError_t err = cudaMalloc((void**)&d_data, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_histogram, size_histogram);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input data to the device.
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // 3. Set 0 to the histogram.
    cudaMemset(d_histogram, 0, size_histogram);

    // 4. Launch the kernel.
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(ceil(N / 1024.0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    aggregationKernel<<<dimGrid, dimBlock>>>(d_data, d_histogram, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 5. Copy the result back to the host.
    cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);

    // 6. Free the device memory.
    cudaFree(d_data);
    cudaFree(d_histogram);

    printf("GPU Time taken: %f ms\n", milliseconds);
}

int main() {
    char data[] = "programming massively parallel processors.";
    unsigned int N = strlen(data);

    // We will use 7 buckets because 26 / 4. 
    unsigned int *histogram_cuda = (unsigned int *)malloc(7 * sizeof(unsigned int));
    printf("Input: %s\n", data);

    histogramCuda(data, histogram_cuda, N);
    printf("Histogram by letter groups:\n");
    for (int i = 0; i < 7; i++) {
        printf("Group %d (%c-%c): %d\n",
            i, 
            'a' + (i * 4), 
            'a' + (i * 4 + 3 < 26 ? i * 4 + 3 : 25), 
            histogram_cuda[i]);
    }

    free(histogram_cuda);
    
    return 0;
}