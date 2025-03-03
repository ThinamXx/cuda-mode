#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define TILE_DIM 8

__constant__ float F[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];

void printMatrix(float *matrix, int width, int height, int depth) {
    printf("\n\n");

    for (int z = 0; z < depth; z++) {
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                printf("%f ", matrix[z * width * height + x * width + y]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n\n");
}

void conv_3D_CPU(
    float *image,
    float *filter,
    float *output,
    int width,
    int height,
    int depth
) {
    // Iterate over each voxel in the output volume.
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                float sum = 0.0f;

                // Apply 3D filter at current position.
                for (int i = 0; i < FILTER_SIZE; i++) {
                    for (int j = 0; j < FILTER_SIZE; j++) {
                        for (int k = 0; k < FILTER_SIZE; k++) {

                            // Calculate input indices with padding offset.
                            int in_row = r + i - FILTER_RADIUS;
                            int in_col = c + j - FILTER_RADIUS;
                            int in_depth = d + k - FILTER_RADIUS;

                            // Boundary check.
                            if (in_row >= 0 && in_row < height && 
                                in_col >= 0 && in_col < width && 
                                in_depth >= 0 && in_depth < depth) {
                                
                                // Calculate correct 3D indices.
                                int img_idx = in_depth * (width * height) + in_row * width + in_col;
                                int filter_idx = i * (FILTER_SIZE * FILTER_SIZE) + j * FILTER_SIZE + k;
                                
                                sum += image[img_idx] * filter[filter_idx];
                            }
                        }
                    }
                }

                // Store result.
                output[d * (width * height) + r * width + c] = sum;
            }
        }
    }
}

__global__ void conv3D_kernel(
    float *image,
    float *output,
    int width,
    int height,
    int depth
) {
    __shared__ float input_tile[TILE_DIM][TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int deep = blockIdx.z * TILE_DIM + threadIdx.z;

    if (row < height && col < width && deep < depth) {
        input_tile[threadIdx.z][threadIdx.y][threadIdx.x] = image[deep * width * height + row * width + col];
    } else {
        input_tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Calculate the output elements. 
    if (row < height && col < width && deep < depth) {
        float sum = 0.0f;

        for (int f_row = 0; f_row < FILTER_SIZE; f_row++) {
            for (int f_col = 0; f_col < FILTER_SIZE; f_col++) {
                for (int f_deep = 0; f_deep < FILTER_SIZE; f_deep++) {
                    if ((int)threadIdx.x - FILTER_RADIUS + f_col >= 0 &&
                        (int)threadIdx.x - FILTER_RADIUS + f_col < TILE_DIM &&
                        (int)threadIdx.y - FILTER_RADIUS + f_row >= 0 &&
                        (int)threadIdx.y - FILTER_RADIUS + f_row < TILE_DIM &&
                        (int)threadIdx.z - FILTER_RADIUS + f_deep >= 0 &&
                        (int)threadIdx.z - FILTER_RADIUS + f_deep < TILE_DIM) {
                        sum += input_tile[threadIdx.z + f_deep][threadIdx.y + f_row][threadIdx.x + f_col] * 
                               F[f_deep][f_row][f_col];
                    }
                    else {
                        if (row - FILTER_RADIUS + f_row >= 0 &&
                            row - FILTER_RADIUS + f_row < height &&
                            col - FILTER_RADIUS + f_col >= 0 &&
                            col - FILTER_RADIUS + f_col < width &&
                            deep - FILTER_RADIUS + f_deep >= 0 &&
                            deep - FILTER_RADIUS + f_deep < depth) {
                            sum += image[(deep - FILTER_RADIUS + f_deep) * width * height + 
                                         (row - FILTER_RADIUS + f_row) * width + 
                                         (col - FILTER_RADIUS + f_col)] * 
                                         F[f_deep][f_row][f_col];
                        }
                    }
                }
            }
        }

        output[deep * width * height + row * width + col] = sum;
    }
}

void conv_3D(
    float *image,
    float *filter,
    float *output,
    int width,
    int height,
    int depth
) {
    int size = width * height * depth * sizeof(float);

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
    cudaMemcpyToSymbol(F, filter, FILTER_SIZE * FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // 2. Launch the kernel to perform the convolution.
    dim3 dimBlock(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 dimGrid(ceil(width / (float)TILE_DIM), ceil(height / (float)TILE_DIM), ceil(depth / (float)TILE_DIM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv3D_kernel<<<dimGrid, dimBlock>>>(d_image, d_output, width, height, depth);
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
    int width = 16;
    int height = 16;
    int depth = 16;
    
    int size = width * height * depth * sizeof(float);
    int filter_size = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE * sizeof(float);

    float *image = (float *)malloc(size);
    float *filter = (float *)malloc(filter_size);
    float *output = (float *)malloc(size);
    float *output_cpu = (float *)malloc(size);

    // Initialize image and filter with random values.
    for (int i = 0; i < width * height * depth; i++) {
        image[i] = rand() % 2;
    }
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; i++) {
        filter[i] = rand() % 2;
    }

    // printMatrix(image, width, height, depth);
    // printMatrix(filter, FILTER_SIZE, FILTER_SIZE, FILTER_SIZE);

    // Call the convolution function. 
    conv_3D(image, filter, output, width, height, depth);
    conv_3D_CPU(image, filter, output_cpu, width, height, depth);

    // printMatrix(output, width, height, depth);

    // Compare the results:
    bool status = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                if (abs(output[k * width * height + i * width + j] - output_cpu[k * width * height + i * width + j]) > 1e-6) {
                    status = false;
                }
            }
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