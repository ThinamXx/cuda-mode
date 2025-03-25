#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BUCKETS 7
#define COARSENING_FACTOR 3

__global__ void histogramKernelCoarsening(char *data, unsigned int *histogram, unsigned int N) {
    // Initialize the privatized bins. 
    __shared__ unsigned int hist_s[NUM_BUCKETS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
        hist_s[bin] = 0u;
    }

    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid * COARSENING_FACTOR; i < min(N, (tid + 1) * COARSENING_FACTOR); i++) {
        int alpha_index = data[i] - 'a';
        if (alpha_index >= 0 && alpha_index < 26) {
            atomicAdd(&(hist_s[alpha_index / 4]), 1);
        }
    }

    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
        unsigned int binValue = hist_s[bin];
        if (binValue >0) {
            atomicAdd(&(histogram[bin]), binValue);
        }
    }
}

__global__ void histogramKernelInterleaved(char *data, unsigned int *histogram, unsigned int N) {
    // Initialize the privatized bins. 
    __shared__ unsigned int hist_s[NUM_BUCKETS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
        hist_s[bin] = 0u;
    }

    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < N; i += blockDim.x * gridDim.x) {
        int alpha_index = data[i] - 'a';
        if (alpha_index >= 0 && alpha_index < 26) {
            atomicAdd(&(hist_s[alpha_index / 4]), 1);
        }
    }

    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
        unsigned int binValue = hist_s[bin];
        if (binValue >0) {
            atomicAdd(&(histogram[bin]), binValue);
        }
    }
}

void histogramCuda(char *data, unsigned int *histogram, unsigned int *histogram_interleaved, unsigned int N) {
    int size = N * sizeof(char);

    int size_histogram = NUM_BUCKETS * sizeof(unsigned int);

    char *d_data;
    unsigned int *d_histogram;
    unsigned int *d_histogram_interleaved;

    // 1. Allocate device memory for the input and output arrays.
    cudaError_t err = cudaMalloc((void**)&d_data, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_histogram, size_histogram);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_histogram_interleaved, size_histogram);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input data to the device.
    err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // 3. Set 0 to the histogram.
    err = cudaMemset(d_histogram, 0, size_histogram);
    err = cudaMemset(d_histogram_interleaved, 0, size_histogram);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 4. Launch the kernel.
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(ceil(N / 1024.0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    histogramKernelCoarsening<<<dimGrid, dimBlock>>>(d_data, d_histogram, N);
    histogramKernelInterleaved<<<dimGrid, dimBlock>>>(d_data, d_histogram_interleaved, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 5. Copy the output data to the host.
    cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogram_interleaved, d_histogram_interleaved, size_histogram, cudaMemcpyDeviceToHost);

    // 6. Free the device memory.
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaFree(d_histogram_interleaved);
    printf("Time taken: %f ms\n", milliseconds);
}

int main() {
    char data[] = "programming massively parallel processors.";
    unsigned int N = strlen(data);

    // We will use 7 buckets because 26 / 4. 
    unsigned int *histogram_cuda = (unsigned int *)malloc(7 * sizeof(unsigned int));
    unsigned int *histogram_cuda_interleaved = (unsigned int *)malloc(7 * sizeof(unsigned int));
    printf("Input: %s\n", data);

    histogramCuda(data, histogram_cuda, histogram_cuda_interleaved, N);
    printf("Histogram by letter groups:\n");
    for (int i = 0; i < 7; i++) {
        printf("Group %d (%c-%c): %d %d\n",
            i, 
            'a' + (i * 4), 
            'a' + (i * 4 + 3 < 26 ? i * 4 + 3 : 25), 
            histogram_cuda[i],
            histogram_cuda_interleaved[i]);
    }

    free(histogram_cuda);
    
    return 0;
}