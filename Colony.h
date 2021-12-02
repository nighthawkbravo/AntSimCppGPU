#pragma once

#include <iostream>
#include "Point.h"
#include "Ant.h"

class Colony {
public:
	Colony(Point p, int ac, int id);
	~Colony();

	void printInfo();
	void printAnts();

	void addAnts(int n);

	inline int getAntCount() { return antCount; }
	inline Point getPos() { return startingPos; }
	inline Ant* getAnts() { return ants; }
	inline int getId() { return myId; }

private:


private:

	int myId;
	int antCount;
	Ant* ants;
	Point startingPos;

};