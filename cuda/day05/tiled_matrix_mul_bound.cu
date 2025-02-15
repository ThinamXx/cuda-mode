#include <stdio.h>
#include <stdlib.h>
#define TILE_SIZE 2

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

__global__ void tiledMatrixMul_kernel(int N, float *A, float *B, float *C) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    int row = by * TILE_SIZE + ty; 
    int col = bx * TILE_SIZE + tx; 

    float sum_val = 0.0f;
    for (int ph = 0; ph < ceil(N / (float)TILE_SIZE); ++ph) {
        // Load A and B tiles into shared memory after checking the bounds.
        if ((row < N) && (ph * TILE_SIZE + tx < N)) {
            A_tile[ty][tx] = A[row * N + ph * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        
        if (((ph * TILE_SIZE + ty) < N) && (col < N)) {
            B_tile[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        // Synchronize the threads to ensure all tiles are loaded. 
        __syncthreads(); 

        // Perform the matrix multiplication. 
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum_val += A_tile[ty][k] * B_tile[k][tx];
        }
        
        // Synchronize the threads to ensure all the multiplications are done
        // for the current phase.
        __syncthreads(); 
    }

    if ((row < N) && (col < N)) {
        C[row * N + col] = sum_val;
    }
}

void tiledMatrixMul(int N, float *A, float *B, float *C) {
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
    dim3 dimGrid(ceil(N / (float)TILE_SIZE), ceil(N / (float)TILE_SIZE), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    tiledMatrixMul_kernel<<<dimGrid, dimBlock>>>(N, A_d, B_d, C_d);
   
    // Part 3: Copy the result back to the host. 
    // free the device memory. 
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 3;
    
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
    tiledMatrixMul(N, A, B, C);

    // Print the result. 
    printMatrix(C, N, N);

    // Free the allocated memory. 
    free(A);
    free(B);
    free(C);
    return 0;
}