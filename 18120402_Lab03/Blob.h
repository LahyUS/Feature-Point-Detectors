#pragma once
class Blob
{
private:
	int _x;
	int _y;
	double _radius;
public:
	int getX() { return _x; }
	int getY() { return _y; }
	int getR() { return _radius; }

	void SetR(int value) { _radius = value; }
	void SetX(int value) { _x = value; }
	void SetY(int value) { _y = value; }

public:
	Blob(int x, int y, double R)
	{
		_x = x;
		_y = y;
		_radius = R;
	}

	Blob(const Blob& other)
	{
		this->_x = other._x;
		this->_y = other._y;
		this->_radius = other._radius;
	}

	Blob()
	{
		this->_x = 0;
		this->_y = 0;
		this->_radius = 0;
	}

	~Blob(){}
};