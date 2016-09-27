#include "TheGameOfLifeWrapper.h"
#include"..\CellUniverse.Models.Unmanaged\TheGameOfLifeNative.h"

bool ** CellUniverse::Models::Unmanaged::Wrappers::CTheGameOfLifeWrapper::
GetNextGeneration() {
	return _impl->GetNextGeneration();
}

void CellUniverse::Models::Unmanaged::Wrappers::CTheGameOfLifeWrapper::
Destroy() {

	if (_impl != nullptr) {
		delete _impl;
		_impl = nullptr;
	}
}

CellUniverse::Models::Unmanaged::Wrappers::CTheGameOfLifeWrapper::
CTheGameOfLifeWrapper(int width, int height)
	: _impl(new Unmanaged::CTheGameOfLifeNative(width, height)) { }

CellUniverse::Models::Unmanaged::Wrappers::CTheGameOfLifeWrapper::
~CTheGameOfLifeWrapper() {
	Destroy();
}

CellUniverse::Models::Unmanaged::Wrappers::CTheGameOfLifeWrapper::
!CTheGameOfLifeWrapper() {
	Destroy();
}