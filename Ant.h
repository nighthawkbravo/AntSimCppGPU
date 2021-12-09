#pragma once

#include "cuda_runtime.h"
#include "Point.h"
#include "direction.h"


class Ant {
public:
	__host__ __device__ Ant();
	__host__ __device__ Ant(Point p);
	__host__ __device__ Ant(const Ant& a) {
		dir = a.dir;
		CarryingFood = a.CarryingFood;
		pos = a.pos;
		colPos = a.colPos;
		lifeSpan = a.lifeSpan;
	}

	const int likly1 = 2;
	const int likly2 = 3;

	void setColPos(Point p);

	__host__ __device__ inline void setPos(Point p) {
		pos = p;
	}
	__host__ __device__ inline void setDir(direction d) {
		dir = d;
	}
	__host__ __device__ inline void setFood() {
		CarryingFood = true;
	}
	__host__ __device__ inline void live() {
		lifeSpan--;
	}

	__host__ __device__ inline direction getdir() { return dir; }
	__host__ __device__ inline bool getCarry() { return CarryingFood; }
	__host__ __device__ inline Point getPos() { return pos; }
	__host__ __device__ inline int getLifeSpan() { return lifeSpan; }

	__host__ __device__ inline direction int2dir(int i) {
		int a = i % 8;

		switch (a) {
		case 0:
			return direction::north;
		case 1:
			return direction::northeast;
		case 2:
			return direction::east;
		case 3:
			return direction::southeast;
		case 4:
			return direction::south;
		case 5:
			return direction::southwest;
		case 6:
			return direction::west;
		case 7:
			return direction::northwest;
		}
	}
	__host__ __device__ inline int dir2int(direction d) {
		switch (d) {
		case direction::north:
			return 0;
		case direction::northeast:
			return 1;
		case direction::east:
			return 2;
		case direction::southeast:
			return 3;
		case direction::south:
			return 4;
		case direction::southwest:
			return 5;
		case direction::west:
			return 6;
		case direction::northwest:
			return 7;
		}
	}

	__host__ __device__ inline direction WillIChangeDirection() {

	}

private:
	
	direction dir;
	bool CarryingFood = false;
	Point pos;
	Point colPos;
	int lifeSpan;

};