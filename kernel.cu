
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Window.h"
#include <stdio.h>
//#include "SDL.h"
#undef main

#define WIDTH 1000
#define HEIGHT 700

// int position, lifespan, isCarryingFood (#, #, 0 or 1)



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//cudaError_t drawWithCuda(SDL_Renderer* r, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void update(uint3 *a) {
    int idx = threadIdx.x;

    a[idx].x = idx;
}

void runUpdate(uint3* a, int size) {
    
    uint3* dev_a;
    cudaMalloc((void**)&dev_a, size * sizeof(uint3));
    
    cudaMemcpy(dev_a, a, size * sizeof(uint3), cudaMemcpyHostToDevice);

    update<<<1, size>>>(dev_a);

    cudaMemcpy(a, dev_a, size * sizeof(uint3), cudaMemcpyDeviceToHost);
}


int main()
{
    Window window("Ant Sim", WIDTH, HEIGHT);
    
    int size = 100;

    uint3* host_ants = new uint3[size];

    for (int i = 0; i < size; ++i) {
        host_ants[i] = { 0,0,0 };
    }

    runUpdate(host_ants, size);
    
    for (int i = 0; i < size; ++i)
        std::cout << host_ants[i].x  << ", " << host_ants[i].y << ", " << host_ants[i].z << std::endl;

    while (!window.isClosed()) {
        window.pollEvents();
        window.clear();
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
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
    addKernel<<< 1, size>>>(dev_c, dev_a, dev_b);

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

//cudaError_t drawWithCuda(SDL_Renderer* r, unsigned int size)
//{
//    SDL_Renderer* dev_r;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);    
//
//    cudaStatus = cudaMalloc((void**)&dev_r, sizeof(dev_r));    
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_r, r, sizeof(dev_r), cudaMemcpyHostToDevice);    
//
//    // Launch a kernel on the GPU with one thread for each element.
//    drawPixel <<< 1, size >> > (r);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();    
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();    
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(r, dev_r, sizeof(dev_r), cudaMemcpyDeviceToHost);    
//
//Error:
//    cudaFree(dev_r);
//
//    return cudaStatus;
//}


//cudaError_t cudaStatus = drawWithCuda(renderer,50);


/*const int arraySize = 5;
        const int a[arraySize] = { 1, 2, 3, 4, 5 };
        const int b[arraySize] = { 10, 20, 30, 40, 50 };
        int c[arraySize] = { 0 };*/

        // Add vectors in parallel.
        /*cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }*/

        /*printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
            c[0], c[1], c[2], c[3], c[4]);*/

            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.
            /*cudaStatus = cudaDeviceReset();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceReset failed!");
                return 1;
            }*/