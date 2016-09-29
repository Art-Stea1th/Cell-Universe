#include "TheGameOfLife.h"
#include"..\CellUniverse.Models.CUDA\TheGameOfLife.cuh"

using namespace CellUniverse::Models::CLI;

bool ** CTheGameOfLife::GetNextGeneration() {
	return _impl->GetNextGeneration();
}

void CTheGameOfLife::Destroy() {

	if (_impl != nullptr) {
		delete _impl;
		_impl = nullptr;
	}
}

CTheGameOfLife::CTheGameOfLife(int width, int height)
	: _impl(new CUDA::CTheGameOfLife(width, height)) { }

CTheGameOfLife::~CTheGameOfLife() {
	Destroy();
}

CTheGameOfLife::!CTheGameOfLife() {
	Destroy();
}