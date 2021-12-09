#include "Colony.h"

Colony::Colony(Point p, int ac, int id) {
	antCount = ac;
	startingPos = p;
	ants = new Ant[ac]; // "Actually in this form you cannot invoke constructor which takes parameter(s). It is not allowed by the language specification. - https://stackoverflow.com/questions/8462895/how-to-dynamically-declare-an-array-of-objects-with-a-constructor-in-c"
	myId = id;

	for (int i = 0; i < antCount; ++i) {
		ants[i].setPos(p);
		ants[i].setColPos(p);
	}
}

Colony::~Colony() {
	delete ants;
}

void Colony::addAnts(int n) {
	/*if (n > 0) {
		int newsize = antCount + n;
		Ant* newAnts = new Ant[newsize];

		for (int i = 0; i < antCount; ++i) {
			newAnts[i] = ants[i];
		}

		for (int i = antCount-1; i < newsize; ++i) {
			newAnts[i].setPos(startingPos);
			newAnts[i].setColPos(startingPos);
		}

		antCount = newsize;
		delete ants;

		ants = newAnts;
	}*/
}

void Colony::setAnts(Ant* a) {
	ants = a;
}

void Colony::printInfo() {
	std::cout << "Colony Id: " << myId
		<< " - Antcount: " << antCount
		<< " - Position: (" << startingPos.getX() << ", " << startingPos.getY() << ")\n";
}

void Colony::printAnts() {
	for (int i = 0; i < antCount; ++i) {
		std::cout << "LifeSpan: " << ants[i].getLifeSpan() << " - "
			<< "Carry: " << ants[i].getCarry() << " - "
			<< "Position: " << ants[i].getPos().toString() << " - "
			<< "Direction: " << dir2string::D2S(ants[i].getdir()) << std::endl;
	}
}
