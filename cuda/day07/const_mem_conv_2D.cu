#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
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
    int width,
    int height,
    int filter_width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float sum = 0.0f;

        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                int image_row = row + i;
                int image_col = col + j;

                float value = (image_row >= 0 && image_row < height && image_col >= 0 && image_col < width) ? image[image_row * width + image_col] : 0.0f;
                sum += value * F[(i + FILTER_RADIUS)][(j + FILTER_RADIUS)];
            }
        }

        output[row * width + col] = sum;
    }
}

void conv_2D(
    float *image,
    float *filter, 
    float *output,
    int width, 
    int height, 
    int filter_width, 
    int filter_height
) {
    int size = width * height * sizeof(float);

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
    cudaMemcpyToSymbol(F, filter, filter_width * filter_height * sizeof(float));

    // 2. Launch the kernel to perform the convolution.
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(width / (float)dimBlock.x), ceil(height / (float)dimBlock.y), 1);
    // size_t size_shared = (dimBlock.x + FILTER_SIZE - 1) * (dimBlock.y + FILTER_SIZE - 1) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2D_kernel<<<dimGrid, dimBlock>>>(d_image, d_output, width, height, filter_width);
    // conv2D_shared_kernel<<<dimGrid, dimBlock, size_shared>>>(d_image, d_output, width, height, filter_width);
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
    int width = 7;
    int height = 7;
    int filter_width = FILTER_SIZE;
    int filter_height = FILTER_SIZE;

    float *image = (float *)malloc(width * height * sizeof(float));
    float *filter = (float *)malloc(filter_width * filter_height * sizeof(float));
    float *output = (float *)malloc(width * height * sizeof(float));

    // Initialize image and filter with random values.
    for (int i = 0; i < width * height; i++) {
        image[i] = rand() % 2;
    }
    for (int i = 0; i < filter_width * filter_height; i++) {
        filter[i] = rand() % 2;
    }

    printMatrix(image, width, height);
    printMatrix(filter, filter_width, filter_height);

    // Call the convolution function. 
    conv_2D(image, filter, output, width, height, filter_width, filter_height);

    printMatrix(output, width, height);

    free(image);
    free(filter);
    free(output);

    return 0;
}
