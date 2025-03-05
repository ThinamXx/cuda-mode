#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

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

void cpu_softmax(float *x, float *y, int seq_len) {
    for (int row = 0; row < seq_len; row++) {

        // 1. Find max value.
        float max_val = x[row * seq_len];
        for (int col = 0; col < seq_len; col++) {
            if (x[row * seq_len + col] > max_val) {
                max_val = x[row * seq_len + col];
            }
        }

        // 2. Compute sum of exp(x[row][col] - max_val).
        float sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            sum += expf(x[row * seq_len + col] - max_val);
        }

        // 3. Compute softmax.
        for (int col = 0; col < seq_len; col++) {
            y[row * seq_len + col] = expf(x[row * seq_len + col] - max_val) / sum;
        }
    }
}

__global__ void softmax_kernel(float *x, float *y, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < seq_len) {
        // 1. Compute max and sum of exp(x[idx][i] - max_val).
        float max_val = -INFINITY;
        float sum_val = 0.0f;

        for (int i = 0; i < seq_len; i++) {
            int cur_val = x[idx * seq_len + i];
            float max_cur = fmaxf(max_val, cur_val);
            float sum_cur = sum_val * expf(max_val - max_cur) + expf(cur_val - max_cur);
            max_val = max_cur;
            sum_val = sum_cur;
        }
        
        // 2. Compute softmax.
        for (int i = 0; i < seq_len; i++) {
            y[idx * seq_len + i] = expf(x[idx * seq_len + i] - max_val) / sum_val;
        }
    }
}

void softmax(float *x, float *y, int seq_len) {
    int size = seq_len * seq_len * sizeof(float);

    float *d_x, *d_y;

    // 1. Allocate device memory for input, output.
    cudaError_t err = cudaMalloc((void**)&d_x, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&d_y, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // 2. Copy input to device memory.
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // 3. Call the kernel to launch the grid of threads.
    dim3 blockDim(256, 1, 1);
    dim3 gridDim(ceil(seq_len / 256.0), 1, 1);
    softmax_kernel<<<gridDim, blockDim>>>(d_x, d_y, seq_len);

    // 4. Copy output from device memory.
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // 5. Free device memory.
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int seq_len = 4;

    float *x = (float *)malloc(seq_len * seq_len * sizeof(float));
    float *y = (float *)malloc(seq_len * seq_len * sizeof(float));
    float *cpu_y = (float *)malloc(seq_len * seq_len * sizeof(float));

    for (int i = 0; i < seq_len * seq_len; i++) {
        x[i] = rand() % 2;
    }

    printMatrix(x, seq_len, seq_len);

    softmax(x, y, seq_len);
    cpu_softmax(x, cpu_y, seq_len);

    printMatrix(y, seq_len, seq_len);
    printMatrix(cpu_y, seq_len, seq_len);

    // Check if the results are the same.
    bool same = true;
    for (int i = 0; i < seq_len * seq_len; i++) {
        if (abs(y[i] - cpu_y[i]) > 1e-6) {
            printf("Error at index %d: %f != %f\n", i, y[i], cpu_y[i]);
            same = false;
            break;
        }
    }
    if (same) {
        printf("Results are the same.\n");
    } else {
        printf("Results are not the same.\n");
    }

    free(x);
    free(y);
    free(cpu_y);
    return 0;
}