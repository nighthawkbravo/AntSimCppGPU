#include "Window.h"

#include <iostream>

Window::Window(const std::string& title, int width, int height) : title(title), width(width), height(height)
{
	closed = !init();
}

Window::~Window() 
{
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	for (auto r : obstacles)
	{
		delete r;
	}
	obstacles.clear();
}

bool Window::init()
{
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cout << "Failed to initalize SDL.\n";
		return 0;
	}

	/*
	SDL_Renderer* renderer;
	SDL_Window* window;
	int i = 0;

	

	
	SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);

	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
	SDL_RenderClear(renderer);
	SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

	for (i = 0; i < WIDTH; ++i)
		SDL_RenderDrawPoint(renderer, i, i);

	SDL_RenderPresent(renderer);

	while (1) {

		SDL_RenderPresent(renderer);


		if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
			break;
	}
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();*/

	SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);	

	if (window == nullptr) {
		std::cerr << "Failed to create window.\n";
		return 0;
	}

	if (renderer == nullptr) {
		std::cerr << "Failed to create renderer.\n";
		return 0;
	}
	
	return true;
}

void Window::pollEvents() {
	SDL_Event event;

	int x = 0, y = 0;

	if (SDL_PollEvent(&event)) {
		if (!obstaclePlacing) {
			switch (event.type) {
			case SDL_QUIT:
				closed = true;
				break;
			case SDL_KEYDOWN:
				switch (event.key.keysym.sym) {
				case SDLK_1:
					std::cout << "Obstacle Placer: The next two mouse clicks will place an obstacle.\n";
					obstaclePlacing = true;
					break;
				}
				break;
			default:
				break;
			}
		}
		else {
			switch (event.type) {
			case SDL_QUIT:
				closed = true;
				break;
			case SDL_MOUSEMOTION:
				if (mouseClickCount == 1) {
					p2x = event.motion.x; p2y = event.motion.y;
				}				
				break;
			case SDL_MOUSEBUTTONDOWN:
				SDL_GetMouseState(&x, &y);
				if (mouseClickCount == 0) {
					mouseClickCount++;
					
					p1x = x; p1y = y;
					std::cout << p1x << ", " << p1y << std::endl;
				}
				else if (mouseClickCount == 1) {
					mouseClickCount = 0;
					p2x = x; p2y = y;
					std::cout << p2x << ", " << p2y << std::endl;
					createRectangle(1);
					obstaclePlacing = false;
				}
				
				break;
			default:
				break;
			}
		}
	}
}

void Window::createRectangle(int c) {
	
	int xUL = 0, yUL = 0;
	int w = 0, h = 0;

	if (p1x < p2x && p1y < p2y) {
		xUL = p1x;
		yUL = p1y;
		w = p2x - p1x;
		h = p2y - p1y;
	}
	else if (p2x < p1x && p2y < p1y) {
		xUL = p2x;
		yUL = p2y;
		w = p1x - p2x;
		h = p1y - p2y;
	}
	else if (p1x < p2x && p1y > p2y) {
		xUL = p1x;
		yUL = p2y;
		w = p2x - p1x;
		h = p1y - p2y;
	}
	else { //(p1X > p2X && p2Y > p1Y) {
		xUL = p2x;
		yUL = p1y;
		w = p1x - p2x;
		h = p2y - p1y;
	}
	
	switch (c) {
	case 1: // Obstacle
		SDL_Rect* rect = new SDL_Rect;
		(*rect).x = xUL; (*rect).y = yUL, (*rect).w = w; (*rect).h = h;
		//(*rect).x = 200; (*rect).y = 200, (*rect).w = 100; (*rect).h = 100;
		std::cout << "Rect: (" << xUL << ", " << yUL << ", " << w << ", " << h << ")\n";
		obstacles.push_back(rect);

	}
}

void Window::clear() const {
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);
	
	if (obstacles.size() > 0) {
		SDL_SetRenderDrawColor(renderer, 165, 42, 42, 255);
		for (auto i = obstacles.begin(); i < obstacles.end(); ++i) {
			SDL_RenderFillRect(renderer, *i);
		}
	}

	SDL_RenderPresent(renderer);
}