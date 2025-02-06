#include <stdio.h>

// Vector addition on the host sum c_h = a_h + b_h.
void vec_add(int n, float* a_h, float* b_h, float* c_h) {
    for (int i = 0; i < n; i++) {
        c_h[i] = a_h[i] + b_h[i];
    }
}

// Vector addition on the device.
// Each thread performs one pair of addition.
__global__ void vec_add_kernel(int n, float* a, float* b, float* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Vector addition on the device.
void vec_add_device(int n, float* a_h, float* b_h, float* c_h) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d; 

    // Part 1: Allocate device memory for a, b, c. 
    // copy a, b to device memory.
    cudaError_t err = cudaMalloc((void**)&a_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&b_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&c_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the vector addition. 
    dim3 dimGrid(ceil(n / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    vec_add_kernel<<<dimGrid, dimBlock>>>(n, a_d, b_d, c_d);

    // Part 3: Copy the result from the device to the host. 
    // Free the device memory. 
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    int N = 10;
    
    // Initialize the vectors. 
    float A[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float B[N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float C[N];

    // Call the vector addition function. 
    // vec_add(N, A, B, C);
    vec_add_device(N, A, B, C);
    // Print the result. 
    for (int i = 0; i < N; i++) {
        printf("%f ", C[i]);
    }

    printf("\n");

    return 0;
}