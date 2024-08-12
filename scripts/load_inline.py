# References:
# https://github.com/pytorch/pytorch/blob/main/test/test_cpp_extensions_jit.py
# https://github.com/cuda-mode/lectures/blob/main/lecture_001/load_inline.py

import torch
from torch.utils.cpp_extension import load_inline


# Define the CUDA kernel and C++ wrapper.
cuda_code = """
__global__ void square_matrix_kernel(const float *matrix, float *result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);
    
    auto result = torch::empty_like(matrix);
    
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x, 
                          (height + threads_per_block.y - 1) / threads_per_block.y);
    
    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    
    return result;
}
"""
cpp_code = "torch::Tensor square_matrix(torch::Tensor matrix);"

# Load the CUDA kernel using load_inline.
square_matrix = load_inline(
    name="square_matrix",
    cpp_sources=cpp_code,
    cuda_sources=cuda_code,
    functions=["square_matrix"],
    with_cuda=True,
    extra_cflags=["-O2"],
    build_directory="./build/load_inline",
)
