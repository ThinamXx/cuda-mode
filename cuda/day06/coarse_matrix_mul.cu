#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define COARSE_FACTOR 4

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

__host__ int calculate_appropriate_tile_size(cudaDeviceProp device_prop) {
    // Refer this implementation: ./dynamic_matrix_mul.cu

    // 1. Shared memory constraints. 
    size_t shared_mem_per_block = device_prop.sharedMemPerBlock;
    size_t max_tile_elements = shared_mem_per_block / (2 * sizeof(float)); // 2 tiles for A and B.
    int tile_size_from_shared = (int)floor(sqrt(max_tile_elements));

    // 2. Thread count constraints. 
    int max_threads_per_block = device_prop.maxThreadsPerBlock;
    int tile_size_from_threads = (int)floor(sqrt(max_threads_per_block));

    // 3. Warp size constraints. 
    int warp_size = device_prop.warpSize;
    
    int tile_size = min(min(tile_size_from_shared, tile_size_from_threads), warp_size);

    // Ensure the tile size is a multiple of the warp size.
    tile_size = (tile_size / warp_size) * warp_size;

    // Print the GPU properties and the tile size.
    printf("Device name: %s\n", device_prop.name);
    printf("Shared memory per block: %zu bytes\n", shared_mem_per_block);
    printf("Max threads per block: %d\n", max_threads_per_block);
    printf("Warp size: %d\n", warp_size);
    printf("Tile size: %d\n", tile_size);
    printf("\n");

    return tile_size;
}

__global__ void coarseMatrixMul_kernel(
    int J, 
    int K, 
    int L, 
    float *A, 
    float *B, 
    float *C,
    int tile_size, 
    unsigned A_tile_offset, 
    unsigned B_tile_offset
) {
    extern __shared__ float A_tile_B_tile[];

    float *A_tile = (float *)(A_tile_B_tile);
    float *B_tile = (float *)(A_tile_B_tile + A_tile_offset/sizeof(float));

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the output matrix C. 
    int row = by * tile_size + ty;
    int colStart = bx * tile_size * COARSE_FACTOR + tx; // each thread block computes COARSE_FACTOR columns of C. 

    // Initialize the sum_value to all output elements. 
    float sum_val[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++) {
        sum_val[i] = 0.0f;
    }

    // Loop over the A and B tiles.
    for (int ph = 0; ph < ceil(K / (float)tile_size); ++ph) {
        if ((row < J) && (ph * tile_size + tx < K)) {
            A_tile[ty * tile_size + tx] = A[row * K + ph * tile_size + tx]; // L is the height and K is the width of the matrix A.
        } else {
            A_tile[ty * tile_size + tx] = 0.0f;
        }

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * tile_size;
            if ((ph * tile_size + ty < K) && (col < L)) {
                B_tile[ty * tile_size + tx] = B[(ph * tile_size + ty) * L + col];
            } else {
                B_tile[ty * tile_size + tx] = 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < tile_size; k++) {
                sum_val[c] += A_tile[ty * tile_size + k] * B_tile[k * tile_size + tx];
            }

            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * tile_size;
        if ((row < J) && (col < L)) {
            C[row * L + col] = sum_val[c];
        }
    }   
}

void coarseMatrixMul(int J, int K, int L, float *A, float *B, float *C) {
    int size_A = J * K * sizeof(float);
    int size_B = K * L * sizeof(float);
    int size_C = J * L * sizeof(float);

    float *A_d, *B_d, *C_d;
    
    // Determine the appropriate tile size. 
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0); // 0 means the first GPU.
    int tile_size = calculate_appropriate_tile_size(device_prop);

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
    
    // Part 2: Launch the kernel to perform the matrix multiplication. 
    dim3 dimBlock(tile_size, tile_size, 1);
    dim3 dimGrid(ceil(J / (float)tile_size), ceil(L / (float)tile_size), 1);
    size_t size = 2 * tile_size * tile_size * sizeof(float); // 2 tiles for A and B.
    coarseMatrixMul_kernel<<<dimGrid, dimBlock, size>>>(J, K, L, A_d, B_d, C_d, tile_size, size/2, size/2);

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
    coarseMatrixMul(J, K, L, A, B, C);

    // Print the result. 
    printMatrix(C, L, J);

    // Free the memory allocated for the matrices. 
    free(A);
    free(B);
    free(C);

    return 0;
}