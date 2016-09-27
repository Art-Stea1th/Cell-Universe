#include <random>
#include "TheGameOfLifeNative.h"

void CellUniverse::Models::Unmanaged::CTheGameOfLifeNative::
Initialize(const int &width, const int &height) {

	_width  = width;
	_height = height;

	_generation = new bool*[_height];

	for (int y(0); y < _height; ++y) {
		_generation[y] = new bool[_width];

		for (int x(0); x < _width; ++x) {
			_generation[y][x] = false;
		}
	}
}

void CellUniverse::Models::Unmanaged::CTheGameOfLifeNative::
Destroy() {

	for (int y(0); y < _height; ++y) {
		delete _generation[y];
	}
	delete _generation;
}

bool ** CellUniverse::Models::Unmanaged::CTheGameOfLifeNative:: // FakeImpl.
GetNextGeneration() {

	std::random_device random;

	for (int y(0); y < _height; ++y) {
		for (int x(0); x < _width; ++x) {
			_generation[y][x] = static_cast<bool>(random() % 2);
		}
	}
	return _generation;
}

CellUniverse::Models::Unmanaged::CTheGameOfLifeNative::
CTheGameOfLifeNative(const int &width, const int &height) {
	Initialize(width, height);
}

CellUniverse::Models::Unmanaged::CTheGameOfLifeNative::
~CTheGameOfLifeNative() {
	Destroy();
}