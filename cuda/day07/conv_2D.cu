#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

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

void conv_2D_CPU(float *image, float *filter, float *output_cpu, int width, int height, int filter_width, int filter_height) {
    for (int row=0; row < height; row++) {
        for (int col=0; col < width; col++) {
            float sum = 0.0f;

            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                    int image_row = row + i;
                    int image_col = col + j;

                    float value = (image_row >= 0 && image_row < height && image_col >= 0 && image_col < width) ? image[image_row * width + image_col] : 0.0f;
                    sum += value * filter[(i + FILTER_RADIUS) * filter_width + (j + FILTER_RADIUS)];
                }
            }
            output_cpu[row * width + col] = sum;
        }
    }
}

__global__ void conv2D_kernel(
    float *image,
    float *filter,
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
                sum += value * filter[(i + FILTER_RADIUS) * filter_width + (j + FILTER_RADIUS)];
            }
        }

        output[row * width + col] = sum;
    }
}

__global__ void conv2D_shared_kernel(
    float *image,
    float *filter,
    float *output,
    int width,
    int height,
    int filter_width
) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = threadIdx.y + FILTER_RADIUS;
    int local_col = threadIdx.x + FILTER_RADIUS;
    int shared_mem_width = blockDim.x + FILTER_SIZE - 1;

    if (row < height && col < width) {
        shared_mem[local_row * shared_mem_width + local_col] = image[row * width + col];
    } else {
        shared_mem[local_row * shared_mem_width + local_col] = 0.0f;
    }

    if (threadIdx.x < FILTER_RADIUS) {
        int left_col = col - FILTER_RADIUS;
        shared_mem[local_row * shared_mem_width + threadIdx.x] = (left_col >= 0) ? image[row * width + left_col] : 0.0f;
    }
    
    if (threadIdx.x >= blockDim.x - FILTER_RADIUS) {
        int right_col = col + FILTER_RADIUS;
        shared_mem[local_row * shared_mem_width + (local_col + FILTER_RADIUS)] = (right_col < width) ? image[row * width + right_col] : 0.0f;
    }

    if (threadIdx.y < FILTER_RADIUS) {
        int top_row = row - FILTER_RADIUS;
        shared_mem[threadIdx.y * shared_mem_width + local_col] = (top_row >= 0) ? image[top_row * width + col] : 0.0f;
    }

    if (threadIdx.y >= blockDim.y - FILTER_RADIUS) {
        int bottom_row = row + FILTER_RADIUS;
        shared_mem[(local_row + FILTER_RADIUS) * shared_mem_width + local_col] = (bottom_row < height) ? image[bottom_row * width + col] : 0.0f;
    }

    __syncthreads();

    if (row < height && col < width) {
        float sum = 0.0f;

        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                sum += shared_mem[(local_row + i) * shared_mem_width + (local_col + j)] * filter[(i + FILTER_RADIUS) * filter_width + (j + FILTER_RADIUS)];
            }
        }

        output[row * width + col] = sum;
    }
}

void conv_2D(float *image, float *filter, float *output, int width, int height, int filter_width, int filter_height) {
    int size = width * height * sizeof(float);
    int filter_size = filter_width * filter_height * sizeof(float);

    float *d_image, *d_filter, *d_output;

    // 1. Allocate device memory for input, filter, and output.
    // Copy input and filter to device memory. 
    cudaError_t err = cudaMalloc((void**)&d_image, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_filter, filter_size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);

    // 2. Launch the kernel to perform the convolution.
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(width / (float)dimBlock.x), ceil(height / (float)dimBlock.y), 1);
    size_t size_shared = (dimBlock.x + FILTER_SIZE - 1) * (dimBlock.y + FILTER_SIZE - 1) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // conv2D_kernel<<<dimGrid, dimBlock>>>(d_image, d_filter, d_output, width, height, filter_width);
    conv2D_shared_kernel<<<dimGrid, dimBlock, size_shared>>>(d_image, d_filter, d_output, width, height, filter_width);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 3. Copy the result back to the host.
    // Free the device memory.
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_filter);
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
    float *output_cpu = (float *)malloc(width * height * sizeof(float));

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
    conv_2D_CPU(image, filter, output_cpu, width, height, filter_width, filter_height);

    printMatrix(output, width, height);
    printMatrix(output_cpu, width, height);

    // Compare the results:
    bool status = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (abs(output[i * width + j] - output_cpu[i * width + j]) > 1e-6) {
                printf("\nError at index %d: %f (GPU) != %f (CPU)\n", i, output[i * width + j], output_cpu[i * width + j]);
                status = false;
            }
        }
    }
    if (status) {
        printf("\nBoth of the outputs are same!!!\n");
    }

    free(image);
    free(filter);
    free(output);
    free(output_cpu);

    return 0;
}