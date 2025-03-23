#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BUCKETS 7

void histogramSeq(char *data, unsigned int *histogram, unsigned int N) {
    for (int i = 0; i < N; i++) {
        int index = data[i] - 'a';
        if (index >= 0 && index < 26) {
            histogram[index / 4]++;
        }
    }
}

__global__ void histogramKernel(char *data, unsigned int *histogram, unsigned int N) {
    // This kernel demonstrates the use of atomic operations (atomicAdd) to update the histogram.
    // Atomic operations are used to ensure that only one thread can update the memory location at a time.
    // This is important when multiple threads are updating the same location in the histogram.

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int alpha_index = data[tid] - 'a';
        if (alpha_index >= 0 && alpha_index < 26) {
            atomicAdd(&(histogram[alpha_index / 4]), 1);
        }
    }
}

void histogramCuda(char *data, unsigned int *histogram, unsigned int N) {
    int size = N * sizeof(char);
    int size_histogram = 7 * sizeof(unsigned int); // 7 buckets with 4 letters each for a-z. 

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
    err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // 3. Launch the kernel.
    dim3 block(1024);
    dim3 grid(ceil(N / (float)block.x));

    histogramKernel<<<grid, block>>>(d_data, d_histogram, N);

    // 4. Copy the output data to the host.
    err = cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_data);
    cudaFree(d_histogram);
    
}

__global__ void histogramKernelPrivate(char *data, unsigned int *histogram, unsigned int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        int alpha_index = data[tid] - 'a';
        if (alpha_index >= 0 && alpha_index < 26) {
            atomicAdd(&(histogram[blockIdx.x * NUM_BUCKETS + alpha_index / 4]), 1);
        }
    }

    if (blockIdx.x > 0) {
        __syncthreads();

        for (unsigned int bin = threadIdx.x; bin < NUM_BUCKETS; bin += blockDim.x) {
            unsigned int binValue = histogram[blockIdx.x * NUM_BUCKETS + bin];
            if (binValue >0) {
                atomicAdd(&(histogram[bin]), binValue);
            }
        }
    }
}

void histogramCudaPrivate(char *data, unsigned int *histogram, unsigned int N) {
    int size = N * sizeof(char);

    // Setup the grid and block dimensions.
    int threads_per_block = 1024;
    int blocks_per_grid = ceil(N / (float)threads_per_block);

    int size_histogram = NUM_BUCKETS * blocks_per_grid * sizeof(unsigned int);

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
    err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // 3. Set 0 to the histogram.
    err = cudaMemset(d_histogram, 0, size_histogram);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 4. Launch the kernel.
    dim3 dimBlock(threads_per_block);
    dim3 dimGrid(blocks_per_grid);

    histogramKernelPrivate<<<dimGrid, dimBlock>>>(d_data, d_histogram, N);

    // 5. Copy the output data to the host.
    cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);

    // 6. Free the device memory.
    cudaFree(d_data);
    cudaFree(d_histogram);
}

int main() {
    char data[] = "programming massively parallel processors.";
    unsigned int N = strlen(data);

    // We will use 7 buckets because 26 / 4. 
    unsigned int *histogram_seq = (unsigned int *)malloc(7 * sizeof(unsigned int));
    unsigned int *histogram_cuda = (unsigned int *)malloc(7 * sizeof(unsigned int));
    unsigned int *histogram_cuda_private = (unsigned int *)malloc(7 * sizeof(unsigned int));
    printf("Input: %s\n", data);

    histogramSeq(data, histogram_seq, N);
    histogramCuda(data, histogram_cuda, N);
    histogramCudaPrivate(data, histogram_cuda_private, N);

    printf("Histogram by letter groups:\n");
    for (int i = 0; i < 7; i++) {
        printf("Group %d (%c-%c): %d %d %d\n",
            i, 
            'a' + (i * 4), 
            'a' + (i * 4 + 3 < 26 ? i * 4 + 3 : 25), 
            histogram_seq[i],
            histogram_cuda[i],
            histogram_cuda_private[i]);
    }

    free(histogram_seq);
    free(histogram_cuda);
    free(histogram_cuda_private);
    
    return 0;
}