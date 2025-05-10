#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void sequentialScanCPU(float *input, float *output, int N) {
    output[0] = input[0];
    for (int idx = 1; idx < N; ++idx) {
        output[idx] = output[idx - 1] + input[idx];
    }
}

void sequentialScan(float *input, float *output, int N) {
    int size = N * sizeof(float);

    float *d_input, *d_output;

    // 1. Allocate device memory for the input and output arrays.
    cudaError_t err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy the input array to the device.
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // 3. Launch the kernel to perform the scan.
    dim3 dimGrid(ceil(N / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    // scanKernel<<<dimGrid, dimBlock>>>(d_input, d_output, N);

    // 4. Copy the result from the device to the host.
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // 5. Free the device memory.
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int N = 7;
    float input[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float output[N];
    float output_cpu[N];

    // sequentialScan(input, output, N);
    sequentialScanCPU(input, output_cpu, N);

    printf("Sequential scan result: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", output_cpu[i]);
    }
    printf("\n");

    return 0;
}