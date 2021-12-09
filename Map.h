#pragma once

class Map {
public:
	Map();
	Map(int w, int h);

	~Map();

	
	void print(); // Avoid printing as this clogs the console.


public:
	int width;
	int height;


	// 1 - Obstacle
	// 2 - Food
	int** grid;

private:
	void zero();
};