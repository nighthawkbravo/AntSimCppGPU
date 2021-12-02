#pragma once
#include <String>

enum class direction { 
	north, 
	northeast, 
	east, 
	southeast, 
	south, 
	southwest, 
	west, 
	northwest
};

static class dir2string {
public:
	static std::string D2S(direction d) {
		switch (d) {
		case direction::north:
			return "north";
		case direction::northeast:
			return "northeast";
		case direction::east:
			return "east";
		case direction::southeast:
			return "southeast";
		case direction::south:
			return "south";
		case direction::southwest:
			return "southwest";
		case direction::west:
			return "west";
		case direction::northwest:
			return "northwest";
		}
	}
};