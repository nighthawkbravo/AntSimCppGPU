#pragma once

#include "cuda_runtime.h"
#include "direction.h"
#include <string>


class Ant {
public:
	Ant();
	Ant(int x, int y);

	void setColPos(int x, int y);
	__host__ __device__ inline void setPos(int x, int y) {
		posX = x;
		posY = y;
	}
	
	__host__ __device__ inline void setFood() {
		CarryingFood = true;
	}

	__host__ __device__ inline direction getdir() { return dir; }
	__host__ __device__ inline bool getLuggage() { return CarryingFood; }
	__host__ __device__ inline int getPosX() { return posX; }
	__host__ __device__ inline int getPosY() { return posY; }
	__host__ __device__ inline int getLifeSpan() { return lifeSpan; }

	inline std::string getPos() {
		std::string s = "(";
		s.append(std::to_string(posX));
		s.append(", ");
		s.append(std::to_string(posY));
		s.append(")");

		return s;
	}


private:
	__host__ __device__ direction int2dir(int i);

private:
	
	direction dir;
	bool CarryingFood = false;
	int posX;
	int posY;

	int colPosX;
	int colPosY;

	int lifeSpan;

};