#include "TheGameOfLifeWrapper.h"
#include"..\CellUniverse.Models.Unmanaged\TheGameOfLifeNative.h"

using namespace CellUniverse::Models::Unmanaged::Wrappers;

bool ** CTheGameOfLifeWrapper::GetNextGeneration() {
	return _impl->GetNextGeneration();
}

void CTheGameOfLifeWrapper::Destroy() {

	if (_impl != nullptr) {
		delete _impl;
		_impl = nullptr;
	}
}

CTheGameOfLifeWrapper::CTheGameOfLifeWrapper(int width, int height)
	: _impl(new CTheGameOfLifeNative(width, height)) { }

CTheGameOfLifeWrapper::~CTheGameOfLifeWrapper() {
	Destroy();
}

CTheGameOfLifeWrapper::!CTheGameOfLifeWrapper() {
	Destroy();
}