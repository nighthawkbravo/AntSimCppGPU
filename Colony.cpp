#include "Colony.h"

Colony::Colony(int x, int y, int ac, int id) {
	antCount = ac;
	posX = x;
	posY = y;
	ants = new Ant[ac]; // "Actually in this form you cannot invoke constructor which takes parameter(s). It is not allowed by the language specification. - https://stackoverflow.com/questions/8462895/how-to-dynamically-declare-an-array-of-objects-with-a-constructor-in-c"
	myId = id;

	for (int i = 0; i < antCount; ++i) {
		ants[i].setPos(posX, posY);
		ants[i].setColPos(posX, posY);
	}
}

Colony::~Colony() {
	delete ants;
}

void Colony::addAnts(int n) {
	if (n > 0) {
		int newsize = antCount + n;
		Ant* newAnts = new Ant[newsize];

		for (int i = 0; i < antCount; ++i) {
			newAnts[i] = ants[i];
		}

		for (int i = antCount-1; i < newsize; ++i) {
			ants[i].setPos(posX, posY);
			ants[i].setColPos(posX, posY);
		}

		antCount = newsize;
		delete ants;

		ants = newAnts;
	}
}

void Colony::setAnts(Ant* a) {
	ants = a;
}

void Colony::printInfo() {
	std::cout << "Colony Id: " << myId
		<< " - Antcount: " << antCount
		<< " - Position: (" << posX << ", " << posY << ")\n";
}

void Colony::printAnts() {
	for (int i = 0; i < antCount; ++i) {
		std::cout << "LifeSpan: " << ants[i].getLifeSpan() << " - "
			<< "Carry: " << ants[i].getLuggage() << " - "
			<< "Position: " << ants[i].getPos() << " - "
			<< "Direction: " << dir2string::D2S(ants[i].getdir()) << std::endl;
	}
}
