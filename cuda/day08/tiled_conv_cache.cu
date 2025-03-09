#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

#define TILE_DIM 32

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

void conv_2D_CPU(float *image, float *filter, float *output, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;

            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    int image_row = row + i - FILTER_RADIUS;
                    int image_col = col + j - FILTER_RADIUS;

                    if (image_row >= 0 && image_row < width && image_col >= 0 && image_col < width) {
                        sum += image[image_row * width + image_col] * filter[i * FILTER_SIZE + j];
                    }
                }
            }

            output[row * width + col] = sum;
        }
    }
}

__global__ void conv2D_kernel(
    float *image,
    float *output,
    int width
) {
    __shared__ float input_tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (row < width && col < width) {
        input_tile[threadIdx.y][threadIdx.x] = image[row * width + col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Calculate the output elements. 
    if (row < width && col < width) {
        float sum = 0.0f;

        for (int f_row = 0; f_row < FILTER_SIZE; f_row++) {
            for (int f_col = 0; f_col < FILTER_SIZE; f_col++) {
                if ((int)threadIdx.x - FILTER_RADIUS + f_col >= 0 &&
                    (int)threadIdx.x - FILTER_RADIUS + f_col < TILE_DIM &&
                    (int)threadIdx.y - FILTER_RADIUS + f_row >= 0 &&
                    (int)threadIdx.y - FILTER_RADIUS + f_row < TILE_DIM) {
                    sum += input_tile[threadIdx.y + f_row][threadIdx.x + f_col] * F[f_row][f_col];
                }
                else {
                    if (row - FILTER_RADIUS + f_row >= 0 &&
                        row - FILTER_RADIUS + f_row < width &&
                        col - FILTER_RADIUS + f_col >= 0 &&
                        col - FILTER_RADIUS + f_col < width) {
                        sum += image[(row - FILTER_RADIUS + f_row) * width + (col - FILTER_RADIUS + f_col)] * F[f_row][f_col];
                    }
                }
            }
        }

        output[row * width + col] = sum;
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
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid(ceil(width / (float)TILE_DIM), ceil(width / (float)TILE_DIM), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2D_kernel<<<dimGrid, dimBlock>>>(d_image, d_output, width);
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
    int N = 7;
    int size = N * N * sizeof(float);
    int filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    float *image = (float *)malloc(size);
    float *filter = (float *)malloc(filter_size);
    float *output = (float *)malloc(size);
    float *output_cpu = (float *)malloc(size);

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
    conv_2D_CPU(image, filter, output_cpu, N);
    printMatrix(output, N, N);

    // Compare the results:
    bool status = true;
    for (int i = 0; i < N * N; i++) {
        if (abs(output[i] - output_cpu[i]) > 1e-6) {
            status = false;
        }
    }
    if (status) {
        printf("\nBoth of the outputs are same!!!\n");
    } else {
        printf("\nBoth of the outputs are not same!!!\n");
    }

    free(image);
    free(filter);
    free(output);
    free(output_cpu);

    return 0;   
}