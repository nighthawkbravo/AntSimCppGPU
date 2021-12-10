#include "Map.h"
#include <iostream>

Map::Map() {
	width = 0;
	height = 0;
	grid = nullptr;
}
Map::Map(int w, int h) {
	width = w;
	height = h;
	grid = new int*[height];
	uniSize = width * height;
	uniGrid = new int[uniSize];

	for (int i = 0; i < height; ++i) {
		grid[i] = new int[width];
	}	
	zero();	
}

Map::~Map() {
	for (int i = 0; i < width; ++i) {
		delete grid[i];
	}
	delete grid;
}

void Map::zero() {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			grid[i][j] = 0;
		}
	}
	for (int i = 0; i < uniSize; ++i) {
		uniGrid[i] = 0;
	}
}

void Map::print() {
	std::cout << "Map Print W:" << width << " H: " << height << std::endl;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) 
			std::cout << grid[i][j];		
		std::cout << std::endl;
	}
}

void Map::print2() {
	std::cout << "Map Print2 US:" << uniSize << std::endl;
	for (int i = 0; i < uniSize; ++i) {
		if (i % width == 0) std::cout << std::endl;
		std::cout << uniGrid[i];
	}
}

void Map::equalizeUniGrid() {
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			uniGrid[j + i * width] = grid[i][j];
}

void Map::equalize2DGrid() {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			grid[i][j] = uniGrid[j + i * width];
		}
	}
}