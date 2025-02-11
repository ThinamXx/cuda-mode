#include <stdio.h>

int main() {
    int device_count;
    cudaDeviceProp device_prop;
    cudaGetDeviceCount(&device_count);
    printf("Number of devices: %d\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaGetDeviceProperties(&device_prop, i);
        printf("Device %d: %s\n", i, device_prop.name);
        printf("Maximum number of threads per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Warp size: %d\n", device_prop.warpSize);
        printf("Registers per block: %d\n", device_prop.regsPerBlock);
        printf("Multi-processor SM count: %d\n", device_prop.multiProcessorCount);
        printf("Clock rate: %d\n", device_prop.clockRate);
        printf("Max threads in x direction: %d\n", device_prop.maxThreadsDim[0]);
        printf("Max threads in y direction: %d\n", device_prop.maxThreadsDim[1]);
        printf("Max threads in z direction: %d\n", device_prop.maxThreadsDim[2]);
        printf("Max grid size in x direction: %d\n", device_prop.maxGridSize[0]);
        printf("Max grid size in y direction: %d\n", device_prop.maxGridSize[1]);
        printf("Max grid size in z direction: %d\n", device_prop.maxGridSize[2]);
        printf("Registers per multiprocessor: %d\n", device_prop.regsPerMultiprocessor);
        printf("Registers per block: %d\n", device_prop.regsPerBlock);   
    }

    return 0;
}