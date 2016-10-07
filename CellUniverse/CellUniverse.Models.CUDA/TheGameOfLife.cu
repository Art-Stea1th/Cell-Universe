#include "TheGameOfLife.cuh"
#include "Universe.cuh"

using namespace CellUniverse::Models::CUDA;

CTheGameOfLife::CTheGameOfLife(const int &width, const int &height) {
	AllocateLinearMatrix(width, height);
	FillRandomLinearMatrix(width, height);
	InitializeDevice(matrix, width, height);
}

void CTheGameOfLife::AllocateLinearMatrix(const int &width, const int &height) {

	matrix = new bool[width * height];

	for (int i(1); i < width; ++i) {
		matrix[i] = matrix[i - 1] + width;
	}
	InitializeLinearMatrix(width, height);
}

void CTheGameOfLife::InitializeLinearMatrix(const int &width, const int &height) {

	int index = width * height;

	while (index--) {
		matrix[index] = false;
	}
}

void CTheGameOfLife::FillRandomLinearMatrix(const int &width, const int &height) {

	int index = width * height;
	std::random_device random;
	std::mt19937 gen(random());

	while (index--) {
		matrix[index] = static_cast<bool>(gen() % 2);		
	}
}

bool* CTheGameOfLife::GetNextGeneration() {
	CalculateNextGeneration(matrix);
	return matrix;
}

CTheGameOfLife::~CTheGameOfLife() {
	FreeDevice();
	DestroyLinearMatrix();
}

void CTheGameOfLife::DestroyLinearMatrix() {
	delete[] matrix; matrix = nullptr;
}