#include <stdio.h>
#include <stdlib.h>

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

__global__ void matrixBlur_kernel(
    int width, 
    int height, 
    unsigned char *in, 
    unsigned char *out
) {
    // Define the BLUR_SIZE. 
    // 2 * BLUR_SIZE + 1 should be equal to the block size or should 
    // give total number of pixels across one dimension of the block.
    // since we are using 3x3 kernel, blur size is 1 because 2 * 1 + 1 = 3.
    const int BLUR_SIZE = 1;

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int pixel_val = 0;
        int pixel_count = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixel_val += in[curRow * width + curCol];
                    ++pixel_count;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixel_val / pixel_count);
    }
}

void matrixBlur(
    int width, 
    int height, 
    unsigned char *in, 
    unsigned char *out
) {
    int size = width * height * sizeof(unsigned char);
    unsigned char *in_d, *out_d;

    // Part 1: Allocate device memory for in and out. 
    // copy in to device memory.
    cudaError_t err = cudaMalloc((void**)&in_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&out_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the matrix blur. 
    dim3 dimGrid(ceil(width / 3.0), ceil(height / 3.0), 1);
    dim3 dimBlock(3, 3, 1);
    matrixBlur_kernel<<<dimGrid, dimBlock>>>(width, height, in_d, out_d);
    
    // Part 3: Copy the result from the device to the host. 
    // free the device memory. 
    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
}

int main() {
    int height = 10;
    int width = 10;

    int size = width * height * sizeof(unsigned char);

    unsigned char *A = (unsigned char *)malloc(size);
    unsigned char *B = (unsigned char *)malloc(size);

    // Initialize the input matrix A with random values. 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int offset = i * width + j;
            A[offset] = rand() % 6; // for values 0-5.
        }
    }
    
    // Print the input matrix. 
    printMatrix(A, width, height);

    // Initialize the kernel. 
    matrixBlur(width, height, A, B);

    // Print the result. 
    printMatrix(B, width, height);

    // Free the allocated memory. 
    free(A);
    free(B);

    return 0;
}