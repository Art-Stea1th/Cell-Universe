#include "TheGameOfLife.h"
#include"..\CellUniverse.Models.Native\Universe.h"

using namespace CellUniverse::Models::CLI;

bool* CTheGameOfLife::GetNextGeneration() {
	return _impl->GetNext();
}

void CTheGameOfLife::Destroy() {

	if (_impl != nullptr) {
		delete _impl;
		_impl = nullptr;
	}
}

CTheGameOfLife::CTheGameOfLife(int width, int height) : _impl(new Native::CPP::CUniverse(width, height)) { }

CTheGameOfLife::~CTheGameOfLife() {
	Destroy();
}

CTheGameOfLife::!CTheGameOfLife() {
	Destroy();
}