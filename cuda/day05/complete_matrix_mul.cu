#include <stdio.h>
#include <stdlib.h>
#define TILE_SIZE 2

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

__global__ void matrixMul_kernel(int J, int K, int L, float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < J) && (col < L)) {
        float c_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            c_sum += A[row * K + k] * B[k * L + col];
        }

        C[row * L + col] = c_sum;
    }
}

void matrixMul(int J, int K, int L, float *A, float *B, float *C) {
    int size_A = J * K * sizeof(float);
    int size_B = K * L * sizeof(float);
    int size_C = J * L * sizeof(float);

    float *A_d, *B_d, *C_d;
    
    // Part 1: Allocate device memory for A, B, and C. 
    // copy A and B to device memory. 
    cudaError_t err = cudaMalloc((void**)&A_d, size_A);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&B_d, size_B);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&C_d, size_C);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(A_d, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_B, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the matrix multiplication. 
    dim3 dimGrid(ceil(J / (float)TILE_SIZE), ceil(L / (float)TILE_SIZE), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    matrixMul_kernel<<<dimGrid, dimBlock>>>(J, K, L, A_d, B_d, C_d); // launch the kernel without shared memory. 

    // Part 3: Copy the result back to the host. 
    // free the device memory. 
    cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int J = 3; // number of rows in matrix A and rows in matrix C.
    int K = 5; // number of columns in matrix A and rows in matrix B.
    int L = 4; // number of columns in matrix B and columns in matrix C.

    int size_A = J * K * sizeof(float);
    int size_B = K * L * sizeof(float);
    int size_C = J * L * sizeof(float);

    float *A = (float *)malloc(size_A);
    float *B = (float *)malloc(size_B);
    float *C = (float *)malloc(size_C);

    // Initialize the matrices A and B with random values. 
    for (int i = 0; i < J; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = rand() % 2;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < L; j++) {
            B[i * L + j] = rand() % 2;
        }
    }
    
    // Print the matrices A and B. 
    printMatrix(A, K, J);
    printMatrix(B, L, K);

    // Call the matrix multiplication function. 
    matrixMul(J, K, L, A, B, C);

    // Print the result. 
    printMatrix(C, L, J);

    // Free the memory allocated for the matrices. 
    free(A);
    free(B);
    free(C);
}


