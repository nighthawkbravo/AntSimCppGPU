﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Colony.h"
#include <vector>
#include "Window.h"
#include <stdio.h>
//#include "SDL.h"
#undef main

#define WIDTH 1000
#define HEIGHT 700

// ---------------------------------------------------------

int id = 0;

__global__ void update(Ant* a);

cudaError_t updateAnts(Ant* host_a, int size);





int main()
{
    Window window("Ant Sim", WIDTH, HEIGHT);
    Ant* a;
    
    Colony c(Point(WIDTH/2+1, HEIGHT/2+1), 10, ++id);

    c.printInfo();
    c.printAnts();

    c.addAnts(2);

    c.printInfo();
    c.printAnts();

    while (!window.isClosed()) {
        window.pollEvents();
        window.clear();
    }

    return 0;
}



__global__ void update(Ant* a) {

}


cudaError_t updateAnts(Colony c) {

    int size = c.getAntCount();
    Ant* host_a = nullptr;

    Ant* dev_a = nullptr;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for the ants vector
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(Ant));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, host_a, size * sizeof(Ant), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to, failed!");
        goto Error;
    }

    update << <1, size >> > (dev_a);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_a, dev_a, size * sizeof(Ant), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy back, failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);

    return cudaStatus;
}




















/*



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void update(uint3 *a) {
    int idx = threadIdx.x;

    a[idx].x = idx;
}

__global__ void update2(Point* a) {
    int idx = threadIdx.x;

    a[idx].setX(idx);
}

void runUpdate(uint3* a, int size) {

    uint3* dev_a;
    cudaMalloc((void**)&dev_a, size * sizeof(uint3));

    cudaMemcpy(dev_a, a, size * sizeof(uint3), cudaMemcpyHostToDevice);

    update<<<1, size>>>(dev_a);

    cudaMemcpy(a, dev_a, size * sizeof(uint3), cudaMemcpyDeviceToHost);
}

void runUpdate2(Point* a, int size) {

    Point* dev_a;
    cudaMalloc((void**)&dev_a, size * sizeof(Point));

    cudaMemcpy(dev_a, a, size * sizeof(Point), cudaMemcpyHostToDevice);

    update2<<<1, size >>> (dev_a);

    cudaMemcpy(a, dev_a, size * sizeof(Point), cudaMemcpyDeviceToHost);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << < 1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

*/