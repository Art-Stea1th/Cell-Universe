#include "TheGameOfLife.cuh"
#include "TheGameOfLifeCUDA.cuh"

using namespace CellUniverse::Models::CUDA;

bool ** CellUniverse::Models::CUDA::CTheGameOfLife::GetNextGeneration() {
	CalculateNextGeneration(next_result);
	return next_result;
}

CTheGameOfLife::CTheGameOfLife(const int &width, const int &height) {
	Initialize(next_result, width, height);
}

CTheGameOfLife::~CTheGameOfLife() {
	Destroy(next_result);
}