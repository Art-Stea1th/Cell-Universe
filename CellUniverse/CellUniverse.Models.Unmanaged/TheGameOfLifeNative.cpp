#include <random>
#include "TheGameOfLifeNative.h"

using namespace std;
using namespace CellUniverse::Models::Unmanaged;


CTheGameOfLifeNative::CTheGameOfLifeNative(const int &width, const int &height) {

	AllocateMatrix(generation, this->width = width, this->height = height);
	AllocateMatrix(buffer2d, this->width, this->height);

	FillRandomMatrix(generation, this->width, this->height);
}

void CTheGameOfLifeNative::AllocateMatrix(bool** &matrix, const int &width, const int &height) {

	matrix = new bool*[height];
	matrix[0] = new bool[height * width];

	for (int i(1); i < height; ++i) {
		matrix[i] = matrix[i - 1] + width;
	}
	InitializeMatrix(matrix, width, height);
}

void CTheGameOfLifeNative::InitializeMatrix(bool** &matrix, const int &width, const int &height) {
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; x++) {
			matrix[y][x] = false;
		}
	}
}

void CTheGameOfLifeNative::FillRandomMatrix(bool** &matrix, const int &width, const int &height) {
	random_device random;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; x++) {
			matrix[y][x] = static_cast<bool>(random() % 2);
		}
	}
}

bool ** CTheGameOfLifeNative::GetNextGeneration() { // temp impl.
	CalculateNextGeneration();
	return generation;
}

bool CTheGameOfLifeNative::TryGetNextGeneration(int &posX, int &posY, bool &value) { // not impl.
	return false;
}

void CTheGameOfLifeNative::CalculateNextGeneration() {

	for (short y = 0; y < height; ++y) {
		for (short x = 0; x < width; ++x) {

			byte neighboursCount = CountNeighbours(generation, x, y);

			if ((neighboursCount == 2 || neighboursCount == 3) && generation[y][x]) {
				buffer2d[y][x] = true;
			}
			if ((neighboursCount < 2 || neighboursCount > 3) && generation[y][x]) {
				buffer2d[y][x] = false;
			}
			if (neighboursCount == 3 && !generation[y][x]) {
				buffer2d[y][x] = true;
			}
		}
	}
	swap(generation, buffer2d);
	InitializeMatrix(buffer2d, width, height);
}

byte CTheGameOfLifeNative::CountNeighbours(bool** &matrix, const int &posX, const int &posY) {

	byte counter = 0;
	int startX = posX - 1, endX = posX + 1;
	int startY = posY - 1, endY = posY + 1;

	for (int y = startY; y <= endY; ++y) {
		for (int x = startX; x <= endX; ++x) {
			if (x == posX && y == posY)
				continue;

			int px = x, py = y;

			if (px == -1) px = width - 1;
			else if (px == width) px = 0;

			if (py == -1) py = height - 1;
			else if (py == height) py = 0;

			if (matrix[py][px])
				counter++;
		}
	}
	return counter;
}

CTheGameOfLifeNative::~CTheGameOfLifeNative() {
	DestroyMatrix(generation);
	DestroyMatrix(buffer2d);
}

void CTheGameOfLifeNative::DestroyMatrix(bool** &matrix) {
	delete[] matrix[0];
	delete[] matrix;
}