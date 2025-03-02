#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

#define INPUT_TILE_SIZE 32
#define OUTPUT_TILE_SIZE (INPUT_TILE_SIZE - 2 * FILTER_RADIUS)

__constant__ float F[FILTER_SIZE][FILTER_SIZE];

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

__global__ void conv2D_kernel(
    float *image,
    float *output,
    int width
) {
    __shared__ float input_tile[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int row = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    if (row >= 0 && row < width && col >= 0 && col < width) {
        input_tile[threadIdx.y][threadIdx.x] = image[row * width + col];
    } else  {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int tile_row = threadIdx.y - FILTER_RADIUS;
    int tile_col = threadIdx.x - FILTER_RADIUS;

    if (row >= 0 && row < width && col >= 0 && col < width) {
        if (tile_row >= 0 && tile_row < OUTPUT_TILE_SIZE && tile_col >= 0 && tile_col < OUTPUT_TILE_SIZE) {
            float sum = 0.0f;

            for (int f_row = 0; f_row < FILTER_SIZE; f_row++) {
                for (int f_col = 0; f_col < FILTER_SIZE; f_col++) {
                    sum += input_tile[tile_row + f_row][tile_col + f_col] * F[f_row][f_col];
                }
            }

            output[row * width + col] = sum;
        }
    }
}

void conv_2D(float *image, float *filter, float *output, int width) {
    int size = width * width * sizeof(float);

    float *d_image, *d_output;

    // 1. Allocate device memory for input, output.
    // Copy input to device memory and filter to constant memory.
    cudaError_t err = cudaMalloc((void**)&d_image, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // 2. Launch the kernel to perform the convolution.
    // Intializing the thread block matrching input tile size.
    dim3 dimBlock(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1);
    dim3 dimGrid(ceil(width / (float)OUTPUT_TILE_SIZE), ceil(width / (float)OUTPUT_TILE_SIZE), 1);

    // // Initializing the thread block matching output tile size.
    // dim3 dimBlockOut(OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE, 1);
    // dim3 dimGridOut(ceil(width / (float)OUTPUT_TILE_SIZE), ceil(width / (float)OUTPUT_TILE_SIZE), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2D_kernel<<<dimGrid, dimBlock>>>(d_image, d_output, width);
    // conv2D_kernel_out<<<dimGridOut, dimBlockOut>>>(d_image, d_output, width);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 3. Copy the result back to the host.
    // Free the device memory.
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);

    printf("GPU time taken: %f milliseconds\n", milliseconds);
}

int main() {
    int N = 16;
    int size = N * N * sizeof(float);
    int filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    float *image = (float *)malloc(size);
    float *filter = (float *)malloc(filter_size);
    float *output = (float *)malloc(size);

    // Initialize image and filter with random values.
    for (int i = 0; i < N * N; i++) {
        image[i] = rand() % 2;
    }
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
        filter[i] = rand() % 2;
    }

    printMatrix(image, N, N);
    printMatrix(filter, FILTER_SIZE, FILTER_SIZE);

    // Call the convolution function. 
    conv_2D(image, filter, output, N);

    printMatrix(output, N, N);

    free(image);
    free(filter);
    free(output);

    return 0;
}