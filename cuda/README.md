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