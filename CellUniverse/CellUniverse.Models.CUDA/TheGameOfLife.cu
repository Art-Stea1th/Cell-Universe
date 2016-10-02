#include "TheGameOfLife.cuh"
#include "TheGameOfLifeCUDA.cuh"

using namespace CellUniverse::Models::CUDA;

CTheGameOfLife::CTheGameOfLife(const int &width, const int &height) {
	AllocateLinearMatrix(width, height);
	FillRandomLinearMatrix(width, height);
	InitializeDevice(matrix, width, height);
}

void CTheGameOfLife::AllocateLinearMatrix(const int &width, const int &height) {

	matrix = new bool*[width];
	matrix[0] = new bool[width * height];

	for (int i(1); i < width; ++i) {
		matrix[i] = matrix[i - 1] + width;
	}
	InitializeLinearMatrix(width, height);
}

void CTheGameOfLife::InitializeLinearMatrix(const int &width, const int &height) {

	int index = width * height;

	while (index--) {
		matrix[0][index] = false;
	}
}

void CTheGameOfLife::FillRandomLinearMatrix(const int &width, const int &height) {

	int index = width * height;
	std::random_device random;
	//std::mt19937 gen(random());
	std::mt19937 gen(time(NULL));

	//srand(time(NULL));

	while (index--) {
		matrix[0][index] = static_cast<bool>(gen() % 2);		
	}
}

bool** CTheGameOfLife::GetNextGeneration() {
	CalculateNextGeneration(matrix);
	return matrix;
}

CTheGameOfLife::~CTheGameOfLife() {
	FreeDevice();
	DestroyLinearMatrix();
}

void CTheGameOfLife::DestroyLinearMatrix() {
	delete[] matrix[0]; matrix[0] = nullptr;
	delete[] matrix; matrix = nullptr;
}