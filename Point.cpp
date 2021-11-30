#include <iostream>
#include <string>

class Point
{
private:
    int x, y;
public:
    Point(int x1, int y1) { x = x1; y = y1; }

    // Copy constructor
    Point(const Point& p1) { x = p1.x; y = p1.y; }

    void print() { 
        std::string s = "X: " + x; 
        s+= " Y: " + y;
        std::cout << s << std::endl; 
    }

    int getX() { return x; }
    int getY() { return y; }
};