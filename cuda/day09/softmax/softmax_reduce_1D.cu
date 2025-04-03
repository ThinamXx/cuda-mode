#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_DIM 32
#define COARSE_FACTOR 4

__global__ void max_kernel(float *input, float *output, int N) {
    __shared__ float shared_data[BLOCK_DIM];
    
    unsigned int segment = COARSE_FACTOR * 2 * BLOCK_DIM * blockIdx.x;
    unsigned int segment_tid = segment + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    if (segment_tid < N) {
        max_val = input[segment_tid];
    }

    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
        max_val = fmaxf(max_val, input[segment_tid + tile * BLOCK_DIM]);
    }

    shared_data[tid] = max_val;
    __syncthreads();

    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(shared_data[0]));
    }
}

__global__ void exp_sum_kernel(float *input, float max_val, float *output, int N) {
    __shared__ float shared_data[BLOCK_DIM];
    
    unsigned int segment = COARSE_FACTOR * 2 * BLOCK_DIM * blockIdx.x;
    unsigned int segment_tid = segment + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float exp_sum_val = 0.0f;
    if (segment_tid < N) {
        exp_sum_val = __expf(input[segment_tid] - max_val);
    }

    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
        unsigned int idx = segment_tid + tile * BLOCK_DIM;
        if (idx < N) {
            exp_sum_val += __expf(input[idx] - max_val);
        }
    }

    shared_data[tid] = exp_sum_val;
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

__global__ void softmax_kernel(float *input, float *output, float max_val, float sum, int N) {
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (tid < N) {
        output[tid] = __expf(input[tid] - max_val) / sum;
    }
}

void softmax(float *input, float *output, int N) {
    float *d_input, *d_output;
    float *d_max, *d_sum;
    float h_max, h_sum;

    // 1. Allocate device memory. 
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    // 2. Copy input to device memory.
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Initialize the max and sum values.
    h_max = -INFINITY;
    h_sum = 0.0f;
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch the kernel to find the max value.
    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(ceil(N / float(BLOCK_DIM * COARSE_FACTOR * 2)), 1, 1);

    max_kernel<<<gridDim, blockDim>>>(d_input, d_max, N);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Compute the sum of exponentials.
    exp_sum_kernel<<<gridDim, blockDim>>>(d_input, h_max, d_sum, N);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Compute the softmax values.
    dim3 normDim(ceil(N / float(BLOCK_DIM)), 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    softmax_kernel<<<normDim, blockDim>>>(d_input, d_output, h_max, h_sum, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Softmax kernel execution time: %f ms\n", milliseconds);

    // 7. Copy the result from the device to the host.  
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 8. Free the device memory.
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max);
    cudaFree(d_sum);
}

int main() {
    int N = 4;
    float input[N] = {6, 7, 8, 3};
    float output[N];

    softmax(input, output, N);

    printf("Softmax output:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Verify with CPU implementation.
    float sum = 0.0f;
    float max_val = input[0];
    
    // Find max value.
    for (int i = 0; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Compute sum of exp(x[i] - max_val).
    for (int i = 0; i < N; i++) {
        sum += expf(input[i] - max_val);
    }
    
    // Compute softmax.
    printf("CPU verification:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", expf(input[i] - max_val) / sum);
    }
    printf("\n");

    return 0;
}
