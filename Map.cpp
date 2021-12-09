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
	grid = new int*[width];

	for (int i = 0; i < width; ++i) {
		grid[i] = new int[height];
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
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			grid[i][j] = 0;
		}
	}
}

void Map::print() {
	std::cout << "Map Print\n";
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			std::cout << grid[i][j];
		}
		std::cout << std::endl;
	}
}