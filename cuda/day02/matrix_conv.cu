#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHANNEL 3

// Create a function to print the matrix. 
void printMatrix(unsigned char *matrix, int width, int height) {
    printf("\n\n");

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

__global__ void matrixConv_kernel(
    int width, 
    int height, 
    unsigned char *P_in,
    unsigned char *P_out
) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        // Get the 1D offset of the matrix. 
        int offset = row * width + col;

        // Convert 1D offset to color image coordinates. 
        int rgb_offset = offset * CHANNEL;
        unsigned char R = P_in[rgb_offset];
        unsigned char G = P_in[rgb_offset + 1];
        unsigned char B = P_in[rgb_offset + 2];

        // Apply the transform. 
        P_out[offset] = 0.21f * R + 0.72f * G + 0.07f * B;
    }
}

void matrixConv(
    int width, 
    int height, 
    unsigned char *P_in, 
    unsigned char *P_out
) {
    int size = width * height * sizeof(unsigned char);
    unsigned char *P_in_d, *P_out_d;
    
    // Part 1: Allocate device memory for P_in and P_out. 
    // copy P_in to device memory.
    cudaError_t err = cudaMalloc((void**)&P_in_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&P_out_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(P_in_d, P_in, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the matrix conversion. 
    dim3 dimGrid(ceil(width / 32.0), ceil(height / 32.0), 1);
    dim3 dimBlock(32, 32, 1);
    matrixConv_kernel<<<dimGrid, dimBlock>>>(width, height, P_in_d, P_out_d);

    // Part 3: Copy the result from the device to the host. 
    // Free the device memory. 
    cudaMemcpy(P_out, P_out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(P_in_d);
    cudaFree(P_out_d);
}

int main() {
    int height = 10;
    int width = 15;

    int size = width * height * sizeof(unsigned char);

    unsigned char *A = (unsigned char *)malloc(size);
    unsigned char *B = (unsigned char *)malloc(size);

    // Initialize the input matrix A with random values. 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int offset = i * width + j;
            A[offset] = rand() % 256;
        }
    }
    
    // Print the input matrix. 
    printMatrix(A, width, height);

    // Initialize the kernel. 
    matrixConv(width, height, A, B);

    // Print the result. 
    printMatrix(B, width, height);

    // Free the allocated memory. 
    free(A);
    free(B);

    return 0;
}