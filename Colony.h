#pragma once

#include <iostream>
#include "Ant.h"

class Colony {
public:
	Colony(int x, int y, int ac, int id);
	~Colony();

	void printInfo();
	void printAnts();

	void addAnts(int n);
	void setAnts(Ant* a);

	inline int getAntCount() { return antCount; }
	inline int getPosX() { return posX; }
	inline int getPosY() { return posY; }
	inline int getId() { return myId; }

	Ant* ants;

private:


private:

	int myId;
	int antCount;
	int posX;
	int posY;

};