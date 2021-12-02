#pragma once

#include "cuda_runtime.h"
#include "Point.h"
#include "direction.h"


class Ant {
public:
	Ant();
	Ant(Point p);

	void setColPos(Point p);
	__host__ __device__ void setPos(Point p);

	__host__ __device__ inline direction getdir() { return dir; }
	__host__ __device__ inline bool getLuggage() { return CarryingFood; }
	__host__ __device__ inline Point getPos() { return pos; }
	__host__ __device__ inline int getLifeSpan() { return lifeSpan; }

	
private:
	__host__ __device__ direction int2dir(int i);

private:
	
	direction dir;
	bool CarryingFood = false;
	Point pos;
	Point colPos;
	int lifeSpan;

};