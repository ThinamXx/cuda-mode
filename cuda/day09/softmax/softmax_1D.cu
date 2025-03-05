#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

// Online softmax kernel that computes max and normalization factor in a single pass.
__global__ void online_softmax_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 1. Compute max and sum of exp(x[idx][i] - max_val).
        float max_val = -INFINITY;
        float sum_val = 0.0f;

        for (int i = 0; i < n; i++) {
            int cur_val = x[idx + i];
            float max_cur = fmaxf(max_val, cur_val);
            float sum_cur = sum_val * expf(max_val - max_cur) + expf(cur_val - max_cur);
            max_val = max_cur;
            sum_val = sum_cur;
        }
        
        // 2. Compute softmax.
        for (int i = 0; i < n; i++) {
            y[idx + i] = expf(x[idx + i] - max_val) / sum_val;
        }
    }
}

void softmax(float *x, float *y, int n) {
    int size = n * sizeof(float);

    float *d_x, *d_y;

    // 1. Allocate device memory for input, output.
    cudaError_t err = cudaMalloc((void**)&d_x, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_y, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy input to device memory.
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // 3. Call the kernel to launch the grid of threads.
    dim3 blockDim(256, 1, 1);
    dim3 gridDim(ceil(n / 256.0), 1, 1);
    online_softmax_kernel<<<gridDim, blockDim>>>(d_x, d_y, n);
    
    // Check for kernel launch errors.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // 4. Copy the result from the device to the host.
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int n = 4;
    float x[n] = {6, 7, 8, 3};
    float y[n];

    softmax(x, y, n);

    printf("Softmax output:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    // Verify with CPU implementation.
    float sum = 0.0f;
    float max_val = x[0];
    
    // Find max value.
    for (int i = 0; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute sum of exp(x[i] - max_val).
    for (int i = 0; i < n; i++) {
        sum += expf(x[i] - max_val);
    }
    
    // Compute softmax.
    printf("CPU verification:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", expf(x[i] - max_val) / sum);
    }
    printf("\n");

    return 0;
}
