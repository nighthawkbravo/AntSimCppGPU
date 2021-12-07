#pragma once

#include <string>
#include <SDL.h>
#include <vector>
#include "Point.h"

class Window {
public:
	Window(const std::string& title, int width, int height);
	~Window();

	void pollEvents();
	void clear() const;

	inline bool isClosed() const { return closed; }

	std::vector<Point*> ants;

private: 
	bool init();
	void createRectangle(int c);
	inline bool validX(int x) const { if (x <= width || x >= 0) return true; return false; }
	inline bool validY(int y) const { if (y <= height || y >= 0) return true; return false; }
	
private:
	std::string title;
	int width, height;

	bool closed = false;
	SDL_Window* window = nullptr;
	SDL_Renderer* renderer = nullptr;
	
	bool obstaclePlacing = false;

	Point p1;
	Point p2;
	int mouseClickCount = 0;

	std::vector<SDL_Rect*> obstacles;	

};