# **cuda** 
This repository contains my notes and exercises from the book mentioned below. 

## **📗Book**
- Programming Massively Parallel Processors.  

## **🛩️Setup Requirements**
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit - Download and install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. A C/C++ compiler (gcc/g++)
4. For VS Code users:
   - C/C++ extension
   - CUDA extension by NVIDIA

## **🚀Progress**

### **Day 01**  
Today, I was focused on setting up the environment and getting familiar with the CUDA programming model. I read the *chapter 1 & 2* of the book and created a simple *hello world* program and *vector addition* program. I learned about cuda function declaration, memory allocation, memory copy, memory free, kernel launch, thread, block, and grid hierarchy in cuda. 

- [x] Setup the environment.
- [x] Create a simple *hello world* program.
- [x] Print a message from the CPU and the GPU.
- [x] Create a program for *vector addition*. 

### **Day 02**
Today, I learned about multidimensional arrays and creating a kernel that performs a matrix transformation. Particularly, I created a kernel that converts a color image to a grayscale image using the given formula: $y = 0.21 \times R + 0.72 \times G + 0.07 \times B$. Similarly, I learned that block and thread labels are (z, y, x) order which is reverse of the usual (x, y, z) order but the grid and block dimensions are specified in the usual (x, y, z) order. 

- [x] Create a kernel that converts a color image to a grayscale image.
- [x] Create a function to print the matrix. 
- [x] Initialize the input matrix with random values. 
- [x] Print the input matrix. 
- [x] Print the result after transformation. 

### **Day 03**
Today, I am still reading the *chapter 3* of the book. I learned how the threads can also interact with each other parallelly within a block. Particularly, I created a kernel where a block of threads computes the average of the pixel using the surrounding pixels including the pixel itself. I also worked on the matrix multiplication kernel where two square matrices are multiplied and the result is stored in a third square matrix. 

- [x] Create a kernel that computes the average using the surrounding and itself.
- [x] Understand the multiple dimensions of threads, blocks, and grids.
- [x] Create a kernel for matrix multiplication. 
- [x] Print the input matrices and the result after multiplication.

### **Day 04**
Today, I focused on the exercises of the *chapter 3* from this book. I have created a kernel that multiplies two matrices in such a way that each thread produces a row of the output matrix i.e. [row_matrix_mul.cu](./exercises/row_matrix_mul.cu). I have also created a kernel that multiplies two matrices in such a way that each thread produces a column of the output matrix i.e. [col_matrix_mul.cu](./exercises/col_matrix_mul.cu). These two kernels are similar to each other but the only difference is the way the threads are indexed and I believe the cons of using these kernels are that we are looping through the matrices multiple times. I also got the chance to work on the matrix-vector multiplication kernel i.e. [matrix_vector_mul.cu](./exercises/matrix_vector_mul.cu).

- [x] Create a kernel that multiplies two matrices with each thread producing a row of the output matrix.
- [x] Create a kernel that multiplies two matrices with each thread producing a column of the output matrix.
- [x] Create a kernel that multiplies a matrix and a vector. 

### **Day 04 [05]**  
I am reading the *chapter 4* of the book. I learned about the streaming multiprocessor (SM), block scheduling, synchronization, and transparent scaling, and warp execution. I also learned about the control divergence, warp scheduling and latency tolerance, resource partitioning, and occupancy, where registers per SM is a limiting factor. I have started reading the *chapter 5* of the book, where I am learning about different memory types and access efficiency in CUDA along with the concept of tiling for reducing the memory access traffic. I have also created a kernel that multiplies two matrices using tiling i.e. [tiled_matrix_mul.cu](./exercises/tiled_matrix_mul.cu).

- [x] Create a host code to get the device properties and device count.
- [x] Create a kernel that multiplies two matrices using tiling. 

### **Day 05**  
I am reading the *chapter 5* of the book, where I am learning to build a tiled matrix multiplication kernel. I have created a kernel that multiplies two matrices using tiling with boundary conditions i.e. [tiled_matrix_mul.cu](./exercises/tiled_matrix_mul.cu) and the code to check the boundary conditions of the tiled matrix, whenever original matrix is not directly divisible by the tile size. I have also created a matrix multiplication kernel that multiplies two matrices with different dimensions i.e. [complete_matrix_mul.cu](./day05/complete_matrix_mul.cu). I have also created a kernel that multiplies two matrices with different dimensions using tiling with boundary conditions i.e. [complete_matrix_mul.cu](./day05/complete_matrix_mul.cu).

- [x] Create a kernel that multiplies two matrices using tiling with boundary conditions.  
- [x] Create a matrix multiplication kernel that multiplies two matrices with different dimensions. 
- [x] Create a matrix multiplication kernel that multiplies two matrices with different dimensions using tiling with boundary conditions.

### **Day 06**
I am working on creating a matrix multiplication kernel that dynamically calculates the tile size based on the GPU properties. I have created a function to calculate the appropriate tile size and a kernel that multiplies two matrices with different dimensions using tiling with boundary conditions [dynamic_matrix_mul.cu](./day06/dynamic_matrix_mul.cu). I am also working on the exercises of the *chapter 5* from this book. I have started reading the *chapter 6* of the book, where I am learning about the memory coalescing and thread coarsening. I have created a kernel that multiplies two matrices using coarsening multiple output tiles [coarse_matrix_mul.cu](./day06/coarse_matrix_mul.cu).

- [x] Create a function to calculate the appropriate tile size using the GPU properties.
- [x] Create a complete kernel that multiplies two matrices with different dimensions using tiling with boundary conditions.
- [x] Create a matrix multiplication kernel with coarsening multiple output tiles.

### **Day 06 [07]**
I am working on the exercises of the *chapter 6* from this book. I have created a kernel that multiplies two matrices using corner turning algorithm where one matrix is transposed while loading from global memory to the shared memory [corner_turning.cu](./day06/corner_turning.cu). I have also created a kernel that convolves a 1D array with a filter [conv_1D.cu](./day07/conv_1D.cu). I have also created a kernel that convolves a 2D array with a filter [conv_2D.cu](./day07/conv_2D.cu) and [const_mem_conv_2D.cu](./day07/const_mem_conv_2D.cu).

- [x] Create a kernel that multiplies two matrices using corner turning algorithm.
- [x] Create a kernel that convolves a 1D array with a filter.
- [x] Create a kernel that convolves a 2D array with a filter using constant memory and shared memory.

### **Day 08**
I am reading the *chapter 7* of the book. I am reading about the parallel convolution, constant memory and caching, and tiled convolution. I have created a kernel that convolves a 2D array with a filter using shared memory and tiling [tiled_conv_2D.cu](./day08/tiled_conv_2D.cu). I have also created a kernel that uses cache for loading the halo elements [tiled_conv_cache.cu](./day08/tiled_conv_cache.cu). Today, I have also worked on integrating the Prewitt filter implemented in CUDA to the PyTorch framework [prewitt/](./day08/prewitt/). The information on building the CUDA extension in PyTorch is:

```bash
cd day08/prewitt
pip install -e .
python prewitt_kernel_test.py
```

- [x] Create a kernel that convolves a 2D array with a filter using shared memory and tiling.
- [x] Create a kernel that convolves a 2D array with a filter using cache for loading the halo elements.
- [x] Integrate the Prewitt filter implemented in CUDA to the PyTorch framework.

### **Day 09**
I am working on creating a CUDA kernel that convolves a 3D array with a filter. I have started working on creating a kernel that computes the softmax of a 1D array [softmax_1D.cu](./day09/softmax/softmax_1D.cu). I have created a CUDA kernel that computes the softmax of a 2D array [softmax_2D.cu](./day09/softmax/softmax_2D.cu) along with the test file [softmax_test.py](./day09/softmax/softmax_test.py) to test the kernel alongside the PyTorch softmax function. I have also created a CUDA kernel that computes the softmax of a 2D array using reduction [softmax_reduce.cu](./day09/softmax/softmax_reduce.cu) and updated the test file [softmax_test.py](./day09/softmax/softmax_test.py) to test the kernel alongside the PyTorch softmax function.

```bash
# RTX 3000 series.
nvcc -arch=sm_86 ....cu -o ...

# RTX 4000 series.
nvcc -arch=sm_89 ....cu -o ...

# For A100.
nvcc -arch=sm_80 ....cu -o ...

# For H100.
nvcc -arch=sm_90 ....cu -o ...
```

- [x] Create a CUDA kernel that convolves a 3D array with a filter.
- [x] Create a CUDA kernel that computes the softmax of a 1D array.
- [x] Create a CUDA kernel that computes the softmax of a 2D array.
- [x] Integrate the softmax implemented in CUDA to the PyTorch framework.
- [x] Create a CUDA kernel that computes the softmax of a 2D array using reduction.

### **Day 10**
I am working on creating a CUDA kernel that performs layer normalization on input matrix of size batch_size x seq_len x embed_dim using shared memory here [layer_norm.cu](./day10/normalization/layer_norm.cu). I strongly believe that I need to learn reduction techniques in CUDA to make the code more efficient. I have created a normalization test file [normalization_test.py](./day10/normalization/normalization_test.py) to test the layer normalization kernel alongside the PyTorch layer normalization function with various configurations and outputs. I have worked on creating a CUDA kernel that performs self-attention here [self_attention.cu](./day10/attention/self_attention.cu). I want to come back to these optimizations later. 

- [x] Create a CUDA kernel that performs layer normalization on input matrix using shared memory.
- [x] Create a test file to test the layer normalization kernel alongside the PyTorch layer normalization function.
- [x] Create a CUDA kernel that performs **self-attention**.

### **Day 11**
I am reading the *chapter 8* of the book where I am learning about the stencil computations. I have created a simple CUDA kernel that performs a stencil computation on a 3D input array [stencil_sweep.cu](./day11/stencil_sweep.cu). I have also created a CUDA kernel that performs a stencil computation on a 3D input array using shared memory [stencil_sweep_shared.cu](./day11/stencil_sweep_shared.cu). I have also created a CUDA kernel that performs a stencil computation on a 3D input array using thread coarsening in the z direction [stencil_sweep_coarse.cu](./day11/stencil_sweep_coarse.cu). In each of these kernels, I have used the constant memory to store the coefficients of the stencil computation. **Thread coarsening** is one of the most important techniques to overcome the limitations of the parallelism with limited resources. 

- [x] Create a CUDA kernel that performs a stencil computation on a 3D input array.
- [x] Create a CUDA kernel that performs a stencil computation on a 3D input array using shared memory.
- [x] Create a CUDA kernel that performs a stencil computation on a 3D input array using thread coarsening in the z direction.

### **Day 12**
I am reading the *chapter 9* of the book where I am learning the concept of histogram computations using atomic operations. I have created a CUDA kernel that performs a histogram computation on a 1D input array using atomic operations [histogram.cu](./day12/histogram.cu). I have also created a kernel that performs histogram privatization where private bins are stored in the shared memory [privatization.cu](./day12/privatization.cu). I have also created a kernel that performs histogram coarsening [coarsening.cu](./day12/coarsening.cu).

- [x] Create a CUDA kernel that performs a histogram computation on a 1D input array using atomic operations.
- [x] Create a CUDA kernel that performs a histogram privatization on a 1D input array using shared memory.
- [x] Create a CUDA kernel that performs a histogram coarsening on a 1D input array.

### **Day 13**
I am reading the **reduction** chapter of the book. The operator needs to be associative *(a + b) + c = a + (b + c)* and commutative *(a + b = b + a)*. I have created a CUDA kernel that performs a sum reduction with optimized memory access on a 1D input array [sum_reduction.cu](./day13/sum_reduction.cu). I have also created a CUDA kernel that performs a sum reduction with shared memory on a 1D input array [sum_reduction_shared.cu](./day13/sum_reduction_shared.cu). I have also created a CUDA kernel that divides the input array into segments and performs a sum reduction on each segment with shared memory [sum_segment_shared.cu](./day13/sum_segment_shared.cu). I have also created the kernel that performs sum reduction with thread coarsening [sum_red_coarse.cu](./day13/sum_red_coarse.cu).

- [x] Create a CUDA kernel that performs a sum reduction on a 1D input array.
- [x] Create a CUDA kernel that performs a sum reduction with shared memory on a 1D input array.
- [x] Create a CUDA kernel that divides the input array into segments and performs a sum reduction on each segment with shared memory.
- [x] Create a CUDA kernel that performs a sum reduction with thread coarsening on a 1D input array.