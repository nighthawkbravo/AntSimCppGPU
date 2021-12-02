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

private: 
	bool init();
	void createRectangle(int c);
	
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