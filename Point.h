#pragma once

#include <iostream>
#include <string>
#include "cuda_runtime.h"

class Point
{
private:
    int x, y;
public:
    __host__ __device__ Point() { x = 0; y = 0; }

    __host__ __device__ Point(int x1, int y1) { x = x1; y = y1; }

    // Copy constructor
    __host__ __device__ Point(const Point& p1) { x = p1.x; y = p1.y; }

    void print();

    __host__ __device__ inline std::string toString() {
        std::string s = "(";
        s.append(std::to_string(x));
        s.append(", ");
        s.append(std::to_string(y));
        s.append(")");

        return s;
    }

    __host__ __device__ int getX() { return x; }
    __host__ __device__ int getY() { return y; }

    __host__ __device__ void setX(int X) { x = X; }
    __host__ __device__ void setY(int Y) { y = Y; }
};