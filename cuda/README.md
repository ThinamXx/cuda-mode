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
I am reading the *chapter 5* of the book, where I am learning to build a tiled matrix multiplication kernel. I have created a kernel that multiplies two matrices using tiling i.e. [tiled_matrix_mul.cu](./exercises/tiled_matrix_mul.cu) and the code to check the boundary conditions of the tiled matrix, whenever original matrix is not directly divisible by the tile size. I have also created a matrix multiplication kernel that multiplies two matrices with different dimensions i.e. [complete_matrix_mul.cu](./day05/complete_matrix_mul.cu). I have also created a kernel that multiplies two matrices with different dimensions using tiling with boundary conditions i.e. [complete_matrix_mul.cu](./day05/complete_matrix_mul.cu).

- [x] Create a kernel that multiplies two matrices using tiling with boundary conditions.  
- [x] Create a matrix multiplication kernel that multiplies two matrices with different dimensions. 
- [x] Create a matrix multiplication kernel that multiplies two matrices with different dimensions using tiling with boundary conditions.

### **Day 06**
I am working on creating a matrix multiplication kernel that dynamically calculates the tile size based on the GPU properties. I have created a function to calculate the appropriate tile size and a kernel that multiplies two matrices with different dimensions using tiling with boundary conditions. Here is the code: [dynamic_matrix_mul.cu](./day06/dynamic_matrix_mul.cu). I am also working on the exercises of the *chapter 5* from this book. I have started reading the *chapter 6* of the book, where I am learning about the memory coalescing and thread coarsening. I have created a kernel that multiplies two matrices using coarsening multiple output tiles. Here is the code: [coarse_matrix_mul.cu](./day06/coarse_matrix_mul.cu).

- [x] Create a function to calculate the appropriate tile size using the GPU properties.
- [x] Create a complete kernel that multiplies two matrices with different dimensions using tiling with boundary conditions.
- [x] Create a matrix multiplication kernel with coarsening multiple output tiles.

### **Day 06 [07]**
I am working on the exercises of the *chapter 6* from this book. I have created a kernel that multiplies two matrices using corner turning algorithm where one matrix is transposed while loading from global memory to the shared memory. Here is the code: [corner_turning.cu](./day06/corner_turning.cu). I have also created a kernel that convolves a 1D array with a filter. Here is the code: [conv_1D.cu](./day07/conv_1D.cu). I have also created a kernel that convolves a 2D array with a filter. Here is the code: [conv_2D.cu](./day07/conv_2D.cu).

- [x] Create a kernel that multiplies two matrices using corner turning algorithm.
- [x] Create a kernel that convolves a 1D array with a filter.
- [x] Create a kernel that convolves a 2D array with a filter.