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

// Create a function to print the vector. 
void printVector(float *vector, int size) {
    printf("\n\n");

    for (int i = 0; i < size; i++) {
        printf("%f ", vector[i]);
    }

    printf("\n\n");
}

__global__ void matrixVectorMul_kernel(int N, float *B, float *C, float *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        for (int j = 0; j < N; j++) {
            A[i] += B[i * N + j] * C[j];
        }
    }
}

void matrixVectorMul(int N, float *B, float *C, float *A) {
    int matrix_size = N * N * sizeof(float);
    int vector_size = N * sizeof(float);

    float *B_d, *C_d, *A_d;

    // Part 1: Allocate device memory for B, C, and A.
    // copy B and C to device memory.
    cudaError_t err = cudaMalloc((void**)&B_d, matrix_size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&C_d, vector_size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&A_d, vector_size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(B_d, B, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, vector_size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the matrix-vector multiplication.
    dim3 dimGrid(ceil(N / 32.0), 1, 1);
    dim3 dimBlock(32, 1, 1);
    matrixVectorMul_kernel<<<dimGrid, dimBlock>>>(N, B_d, C_d, A_d);

    // Part 3: Copy the result from the device to the host. 
    // free the device memory. 
    cudaMemcpy(A, A_d, vector_size, cudaMemcpyDeviceToHost);

    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(A_d);
}

int main() {
    int N = 3;
    int matrix_size = N * N * sizeof(float);
    int vector_size = N * sizeof(float);

    float *B = (float *)malloc(matrix_size);
    float *C = (float *)malloc(vector_size);
    float *A = (float *)malloc(vector_size);
    
    // Initialize the matrix and vector. 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            B[offset] = rand() % 2;
        }
    }
    for (int i = 0; i < N; i++) {
        C[i] = rand() % 2;
    }

    // Print the matrix and vector. 
    printMatrix(B, N, N);
    printVector(C, N);

    // Call the matrix-vector multiplication function. 
    matrixVectorMul(N, B, C, A);

    // Print the result. 
    printVector(A, N);

    // Free the allocated memory. 
    free(B);
    free(C);
    free(A);

    return 0;
}