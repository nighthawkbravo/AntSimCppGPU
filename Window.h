#pragma once

#include <string>
#include <SDL.h>
#include <vector>
#include "Point.h"
#include "Map.h"

class Window {
public:
	Window(const std::string& title, int width, int height, Map* _m);
	~Window();

	void cleanAnts();
	void pollEvents();
	void clear() const;
	void setTitle(const char* title);
	inline bool isClosed() const { return closed; }
	

public:
	std::vector<Point*> ants;
	bool pause = true;

private: 
	bool init();
	void createRectangle(int c);
	inline bool validX(int x) const { if (x <= width || x >= 0) return true; return false; }
	inline bool validY(int y) const { if (y <= height || y >= 0) return true; return false; }
	void fillMapWithRect(int ULx, int ULy, int w, int h, int value);
	
private:
	std::string title;
	int width, height;
	Map* m;

	bool closed = false;
	SDL_Window* window = nullptr;
	SDL_Renderer* renderer = nullptr;
	
	bool obstaclePlacing = false;

	Point p1;
	Point p2;
	int mouseClickCount = 0;

	std::vector<SDL_Rect*> obstacles;	

};