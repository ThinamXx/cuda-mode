#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");

    // Launch kernel with a single thread block
    helloFromGPU<<<1, 6>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}