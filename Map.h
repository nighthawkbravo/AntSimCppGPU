#pragma once

class Map {
public:
	Map();
	Map(int w, int h);

	~Map();

	void print();


public:
	int width;
	int height;
	int** grid;

private:
	void zero();
};