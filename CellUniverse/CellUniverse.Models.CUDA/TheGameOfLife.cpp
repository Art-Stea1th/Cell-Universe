#include "TheGameOfLife.h"
#include <random>

using namespace std;
using namespace CellUniverse::Models::CUDA;


CTheGameOfLife::CTheGameOfLife(const int &width, const int &height) {

	AllocateLinearMatrix(generation, this->width = width, this->height = height);
	AllocateLinearMatrix(buffer, this->width, this->height);

	FillRandomLinearMatrix(generation, this->width, this->height);
}

void CTheGameOfLife::AllocateLinearMatrix(bool** &matrix, const int &width, const int &height) {

	matrix = new bool*[height];
	matrix[0] = new bool[height * width];

	for (int i(1); i < height; ++i) {
		matrix[i] = matrix[i - 1] + width;
	}
	InitializeLinearMatrix(matrix, width, height);
}

void CTheGameOfLife::InitializeLinearMatrix(bool** &matrix, const int &width, const int &height) {

	int index = width * height;

	while (index--) {
		matrix[0][index] = false;
	}
}

void CTheGameOfLife::FillRandomLinearMatrix(bool** &matrix, const int &width, const int &height) {

	int index = width * height; random_device random;

	while (index--) {
		matrix[0][index] = static_cast<bool>(random() % 2);
	}
}

bool ** CTheGameOfLife::GetNextGeneration() { // temp impl.
	CalculateNextGeneration();
	return generation;
}

bool CTheGameOfLife::TryGetNextGeneration(int &posX, int &posY, bool &value) { // not impl.
	return false;
}

void CTheGameOfLife::CalculateNextGeneration() {

	for (short y = 0; y < height; ++y) {
		for (short x = 0; x < width; ++x) {

			byte neighboursCount = CountNeighbours(generation, x, y);

			if ((neighboursCount == 2 || neighboursCount == 3) && generation[y][x]) {
				buffer[y][x] = true;
			}
			if ((neighboursCount < 2 || neighboursCount > 3) && generation[y][x]) {
				buffer[y][x] = false;
			}
			if (neighboursCount == 3 && !generation[y][x]) {
				buffer[y][x] = true;
			}
		}
	}
	swap(generation, buffer);
	InitializeLinearMatrix(buffer, width, height);
}

byte CTheGameOfLife::CountNeighbours(bool** &matrix, const int &posX, const int &posY) {

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

CTheGameOfLife::~CTheGameOfLife() {
	DestroyLinearMatrix(generation);
	DestroyLinearMatrix(buffer);
}

void CTheGameOfLife::DestroyLinearMatrix(bool** &matrix) {
	delete[] matrix[0];
	delete[] matrix;
}