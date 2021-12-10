#pragma once

class Map {
public:
	Map();
	Map(int w, int h);

	~Map();

	
	void print(); // Avoid printing as this clogs the console.
	void print2();
	void equalizeUniGrid();
	void equalize2DGrid();
	

public:
	int width;
	int height;
	int uniSize;


	// 1 - Obstacle
	// 2 - Food
	int** grid;
	int* uniGrid;

private:
	void zero();
};