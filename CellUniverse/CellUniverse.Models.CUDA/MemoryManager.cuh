#pragma once

#include "DefinesCUDA.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			template <typename T> HT void AllocateVectorOnHost(T* &hPtr, int vectorLength);
			template <typename T> HT void InitializeVectorOnHost(T* &hPtr, int vectorLength);
			template <typename T> HT void FreeVectorOnHost(T* hPtr);

			template <typename T> HT void AllocateVectorOnDevice(T* &dPtr, int alignedBlocksCount, int threadsPerBlock);
			template <typename T> GL void InitializeVectorOnDevice(T* &dPtr, int alignedBlocksCount, int threadsPerBlock);
			template <typename T> HT void FreeVectorOnDevice(T* dPtr);

		}
	}
}
#include "MemoryManager.cu"