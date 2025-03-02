#include <math.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define FILTER_RADIUS 1
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define TILE_DIM 32

__constant__ float PREWITT_X[FILTER_SIZE][FILTER_SIZE];
__constant__ float PREWITT_Y[FILTER_SIZE][FILTER_SIZE];

__global__ void prewitt_kernel(
    float *image,
    float *output,
    int width,
    int height
) {
    __shared__ float input_tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (row < height && col < width) {
        input_tile[threadIdx.y][threadIdx.x] = image[row * width + col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (row < height && col < width) {
        float sum_x = 0.0f;
        float sum_y = 0.0f;

        for (int f_row = 0; f_row < FILTER_SIZE; f_row++) {
            for (int f_col = 0; f_col < FILTER_SIZE; f_col++) {
                if ((int)threadIdx.x - FILTER_RADIUS + f_col >= 0 &&
                    (int)threadIdx.x - FILTER_RADIUS + f_col < TILE_DIM &&
                    (int)threadIdx.y - FILTER_RADIUS + f_row >= 0 &&
                    (int)threadIdx.y - FILTER_RADIUS + f_row < TILE_DIM) {
                    sum_x += input_tile[threadIdx.y + f_row][threadIdx.x + f_col] * PREWITT_X[f_row][f_col];
                    sum_y += input_tile[threadIdx.y + f_row][threadIdx.x + f_col] * PREWITT_Y[f_row][f_col];
                } 
                else {
                    if (row - FILTER_RADIUS + f_row >= 0 &&
                        row - FILTER_RADIUS + f_row < height &&
                        col - FILTER_RADIUS + f_col >= 0 &&
                        col - FILTER_RADIUS + f_col < width) {
                        sum_x += image[(row - FILTER_RADIUS + f_row) * width + (col - FILTER_RADIUS + f_col)] * PREWITT_X[f_row][f_col];
                        sum_y += image[(row - FILTER_RADIUS + f_row) * width + (col - FILTER_RADIUS + f_col)] * PREWITT_Y[f_row][f_col];
                    }
                }
            }
        }

        output[row * width + col] = sqrtf(sum_x * sum_x + sum_y * sum_y);
    }
}

void prewitt_filters() {
    float prewitt_x[FILTER_SIZE][FILTER_SIZE] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    float prewitt_y[FILTER_SIZE][FILTER_SIZE] = {
        {-1, -1, -1},
        { 0,  0,  0},
        { 1,  1,  1}
    };

    cudaMemcpyToSymbol(PREWITT_X, prewitt_x, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMemcpyToSymbol(PREWITT_Y, prewitt_y, FILTER_SIZE * FILTER_SIZE * sizeof(float));
}

torch::Tensor prewitt_cuda_forward(torch::Tensor input) {
    input = input.contiguous();

    const int height = input.size(0);
    const int width = input.size(1);

    auto output = torch::zeros_like(input);

    const dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    const dim3 dimGrid(ceil(width / (float)TILE_DIM), ceil(height / (float)TILE_DIM), 1);

    prewitt_filters();

    prewitt_kernel<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width,
        height
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error in prewitt_cuda_forward: %s\n", cudaGetErrorString(err));
    }

    return output;
}