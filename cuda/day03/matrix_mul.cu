#include <stdio.h>
#include <stdlib.h>

// Create a function to print the matrix. 
void printMatrix(float *matrix, int width, int height) {
    printf("\n\n");

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

__global__ void matrixMul_kernel(int N, float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < N) && (col < N)) {
        float c_sum = 0.0f;
        // Perform the matrix multiplication
        // using the row-major order. 
        for (int k = 0; k < N; k++) {
            c_sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = c_sum;
    }    
}

void matrixMul(int N, float *A, float *B, float *C) {
    int size = N * N * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B, and C.
    // copy A and B to device memory. 
    cudaError_t err = cudaMalloc((void**)&A_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&B_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&C_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the matrix multiplication. 
    dim3 dimGrid(ceil(N / 32.0), ceil(N / 32.0), 1);
    dim3 dimBlock(32, 32, 1);
    matrixMul_kernel<<<dimGrid, dimBlock>>>(N, A_d, B_d, C_d);

    // Part 3: Copy the result back to the host. 
    // free the device memory. 
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 5;
    
    int size = N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    // Initialize the matrices. 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            A[offset] = rand() % 2;
            B[offset] = rand() % 2;
        }
    }

    // Print the matrices. 
    printMatrix(A, N, N);
    printMatrix(B, N, N);

    // Call the matrix multiplication function. 
    matrixMul(N, A, B, C);

    // Print the result. 
    printMatrix(C, N, N);

    // Free the allocated memory. 
    free(A);
    free(B);
    free(C);
    return 0;
}

