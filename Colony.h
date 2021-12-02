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
	void setAnts(Ant* a);

	inline int getAntCount() { return antCount; }
	inline Point getPos() { return startingPos; }
	inline int getId() { return myId; }

	Ant* ants;

private:


private:

	int myId;
	int antCount;
	Point startingPos;

};