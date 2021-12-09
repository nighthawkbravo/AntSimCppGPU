#include "Ant.h"

__host__ __device__ Ant::Ant() {
	pos = Point(0, 0);
	dir = int2dir(rand() % 8); // 0 - 7
	lifeSpan = (rand() % 200 + 1) * 40; // 40 - 8000 ticks
}

__host__ __device__ Ant::Ant(Point p) {
	pos = p;
	colPos = p;
	dir = int2dir(rand() % 8); // 0 - 7
	lifeSpan = (rand() % 200 + 1) * 40; // 40 - 8000 ticks
}

void Ant::setColPos(Point p) {
	colPos = p;
}

//__host__ __device__ direction Ant::int2dir(int i) {
//	int a = i % 8;
//
//	switch (a) {
//	case 0:
//		return direction::north;
//	case 1:
//		return direction::northeast;
//	case 2:
//		return direction::east;
//	case 3:
//		return direction::southeast;
//	case 4:
//		return direction::south;
//	case 5:
//		return direction::southwest;
//	case 6:
//		return direction::west;
//	case 7:
//		return direction::northwest;
//	}
//}

//__host__ __device__ int Ant::dir2int(direction d) {
//	switch (d) {
//	case direction::north:
//		return 0;
//	case direction::northeast:
//		return 1;
//	case direction::east:
//		return 2;
//	case direction::southeast:
//		return 3;
//	case direction::south:
//		return 4;
//	case direction::southwest:
//		return 5;
//	case direction::west:
//		return 6;
//	case direction::northwest:
//		return 7;
//	}
//}