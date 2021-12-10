
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <thread>
#include <chrono>


#include <algorithm>
#include <iterator>
#include <vector>

#include "Window.h"
#include "Colony.h"
#include "Map.h"
#include "direction.h"


#include <stdio.h>
#include <string> 
#include <sstream>
//#include "SDL.h"
#undef main


#define WIDTH 1500
#define HEIGHT 700




#define THREADSPERBLOCK 1024

// ---------------------------------------------------------

int startX = WIDTH / 2 + 1;
int startY = HEIGHT / 2 + 1;

int numOfAnts = 10000;

//int startX = 100;
//int startY = 100;

int id = 0;
Window* win;

__device__ int generate(curandState* globalState, int ind);
__device__ int clampX(int x);
__device__ int clampY(int y);
__global__ void setup_kernel(curandState* state, unsigned long seed, int size);
__global__ void update(Ant* a, curandState* globalState, int* dev_map, int w, int h, int size);
cudaError_t updateAnts(Colony *c, Window* w);

int frameCount, timerFPS, lastFrame, fps;
int lastTime;

int main()
{
    Map* m = new Map(WIDTH, HEIGHT);
    
    std::string title = "ANT SIM GPU - Ticks: ";

    int tickCount = 0;

    win = new Window("Ant Sim", WIDTH, HEIGHT, m);
    std::ostringstream strs;
    strs << tickCount;    
    win->setTitle(title.append(strs.str()).c_str());


    Colony c(Point(startX, startY), numOfAnts, ++id);
    
    c.printInfo();

    while (!win->isClosed()) {
        auto start = std::chrono::high_resolution_clock::now();

        win->pollEvents();
        win->clear();
        if (!win->pause) {
            updateAnts(&c, win);
            tickCount++;
            if (tickCount % 15 == 0)
            {
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

                double d;
                if (duration.count() != 0) {
                    d = (1.0 / duration.count()) * 1000;
                    //std::cout << "D: " << duration.count() << std::endl;
                }
                else d = 1;

                std::ostringstream strs;
                strs << d;
                title = "ANT SIM GPU - Ticks: ";
                win->setTitle(title.append(strs.str()).c_str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        //c.printAnts();
        

               
    }

    return 0;
}


// --------------- Ant Cuda ---------------

__device__ direction goHome(Point p, Point c) {
    int px = p.getX();
    int py = p.getY();

    int cx = c.getX();
    int cy = c.getY();

    if (py < cy && px == cx) return direction::north;
    if (py < cy && px < cx) return direction::northeast;
    if (py == cy && px < cx) return direction::east;
    if (py > cy && px < cx) return direction::southeast;
    if (py > cy && px == cx) return direction::south;
    if (py > cy && px > cx) return direction::southwest;
    if (py == cy && px > cx) return direction::west;
    return direction::northwest;

}

__host__ __device__ int XY2UNI(int x, int y) {
    return x + y * WIDTH;
}

__device__ Point move(direction d, Point p, int* m) {
    
    int x = p.getX();
    int y = p.getY();

    switch (d) {
    case direction::north:
        if (m[XY2UNI(clampX(x), clampY(y + 1))] != 1)
            return Point(clampX(x), clampY(y + 1));
        return p;
    case direction::northeast:
        if (m[XY2UNI(clampX(x + 1), clampY(y + 1))] != 1)
            return Point(clampX(x + 1), clampY(y + 1));
        return p;
    case direction::east:
        if (m[XY2UNI(clampX(x + 1), clampY(y))] != 1)
            return Point(clampX(x + 1), clampY(y));
        return p;
    case direction::southeast:
        if (m[XY2UNI(clampX(x + 1), clampY(y - 1))] != 1)
            return Point(clampX(x + 1), clampY(y - 1));
        return p;
    case direction::south:
        if (m[XY2UNI(clampX(x), clampY(y - 1))] != 1)
            return Point(clampX(x), clampY(y - 1));
        return p;
    case direction::southwest:
        if (m[XY2UNI(clampX(x - 1), clampY(y - 1))] != 1)
            return Point(clampX(x - 1), clampY(y - 1));
        return p;
    case direction::west:
        if (m[XY2UNI(clampX(x - 1), clampY(y))] != 1)
            return Point(clampX(x - 1), clampY(y));
        return p;
    case direction::northwest:
        if (m[XY2UNI(clampX(x - 1), clampY(y + 1))] != 1)
            return Point(clampX(x - 1), clampY(y + 1));
        return p;
    }
    
    
    
}
__device__ int generate(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];    
    int RANDOM = curand(&localState) % 3 - 1; // -1, 0, 1
    //int RANDOM = curand(&localState) % 5 - 2; // -2, -1, 0, 1, 2
    globalState[ind] = localState;
    return RANDOM;
}
__device__ int generate2(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];
    int RANDOM = curand(&localState) % 7; // [0, 6]
    //int RANDOM = 0;
    globalState[ind] = localState;
    return RANDOM;
}

__device__ int clampX(int x) {
    if (x < 0) return 0;
    if (x > WIDTH) return WIDTH;
    return x;
}
__device__ int clampY(int y) {
    if (y < 0) return 0;
    if (y > HEIGHT) return HEIGHT;
    return y;
}

__global__ void setup_kernel(curandState* state, unsigned long seed, int size)
{
    //int id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) return;
    curand_init(seed, id, 0, &state[id]);
}
__global__ void update(Ant* a, curandState* globalState, int* dev_map, int w, int h, int size) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    if (a[idx].getLifeSpan() > 0) {
        int r;

        if (dev_map[XY2UNI(a[idx].getPos().getX(), a[idx].getPos().getY())] == 2 && !a[idx].getCarry()) a[idx].setFood(1);
        if (a[idx].getCarry() && a[idx].getPos() == a[idx].getColPos()) a[idx].setFood(0);

        if (a[idx].getCarry()) r = a[idx].likly1;
        else r = a[idx].likly2;

        int n2 = generate2(globalState, idx);

        if (n2 < r) {
            int intdir = a[idx].dir2int(a[idx].getdir()) + 8;
            int n = generate(globalState, idx);           

            a[idx].setDir(a[idx].int2dir(intdir + n));
        }

        if (a[idx].getCarry()) a[idx].setDir(goHome(a[idx].getPos(), a[idx].getColPos()));
        
        a[idx].setPos(move(a[idx].getdir(), a[idx].getPos(), dev_map));
        a[idx].live();
    }
}
cudaError_t updateAnts(Colony *c, Window* w) {
    
    w->cleanAnts();
    int size = c->getAntCount();

    Ant* dev_ants;
    int* dev_uniGrid;
    curandState* devStates;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU buffers for the ants vector
    cudaStatus = cudaMalloc((void**)&dev_ants, size * sizeof(Ant));
    cudaStatus = cudaMalloc(&devStates, size * sizeof(curandState));    
    cudaStatus = cudaMalloc((void**)&dev_uniGrid, w->m->uniSize * sizeof(int));



    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ants, c->ants, size * sizeof(Ant), cudaMemcpyHostToDevice); 
    cudaStatus = cudaMemcpy(dev_uniGrid, w->m->uniGrid, w->m->uniSize * sizeof(int), cudaMemcpyHostToDevice);
    
    srand(time(0));
    int seed = rand();

    int noBlocks = size / THREADSPERBLOCK + 1;

    if (size <= 1024) {
        setup_kernel<<<1, size>>>(devStates, seed, size);

        update<<<1, size>>>(dev_ants, devStates, dev_uniGrid, WIDTH, HEIGHT, size);
    }
    if (size > 1024) {
        setup_kernel<<<noBlocks, 1024>>>(devStates, seed, size);

        update<<<noBlocks, 1024>>>(dev_ants, devStates, dev_uniGrid, WIDTH, HEIGHT, size);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();    

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c->ants, dev_ants, size * sizeof(Ant), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(w->m->uniGrid, dev_uniGrid, w->m->uniSize * sizeof(int), cudaMemcpyDeviceToHost); // For perhaps later

    
Error:
    cudaFree(dev_ants);
    cudaFree(devStates);
    cudaFree(dev_uniGrid);


    int* keys = new int[size];
    Point* values = new Point[size];

    for (int i = 0; i < size; ++i) {
        if (c->ants[i].getLifeSpan() > 0) {
            w->ants.push_back(new Point(c->ants[i].getPos()));

            keys[i] = XY2UNI(c->ants[i].getPos().getX(), c->ants[i].getPos().getY());
            values[i] = c->ants[i].getPos();
        }
    }
    
    
    thrust::stable_sort_by_key(keys, keys + size, values);

    std::vector<Point> points(values, values + size);

    points.erase(std::unique(points.begin(), points.begin()+size));
    w->uniqueAnts = points;

    return cudaStatus;
}













/*

cudaError_t updateAnts(Colony *c) {

    int size = c->getAntCount();

    Ant* dev_ants;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for the ants vector
    cudaStatus = cudaMalloc((void**)&dev_ants, size * sizeof(Ant));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ants, c->ants, size * sizeof(Ant), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to, failed!");
        goto Error;
    }

    update<<<1, size>>>(dev_ants);

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
    cudaStatus = cudaMemcpy(c->ants, dev_ants, size * sizeof(Ant), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy back, failed!");
        goto Error;
    }

Error:
    cudaFree(dev_ants);

    return cudaStatus;
}

*/

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



/*
Useful Links

https://stackoverflow.com/questions/36274112/generate-random-number-within-cuda-kernel - Random numbers for each thread;

*/