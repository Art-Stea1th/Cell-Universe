#pragma once
#include "DefinesCUDA.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			GL void Next(bool* dVector, bool* dBuffer, int width, int height);
			GL void SwapValues(bool* dVector, bool* dBuffer);
			GL void InitializeVectorOnDevice(bool* dVector, int alignedBlocksCount, int threadsPerBlock);

			HT void AllocateVectorOnDevice(bool* &dVector, int alignedBlocksCount, int threadsPerBlock);
			GL void InitializeVectorOnDevice(bool* dVector, int alignedLength);
			HT void FreeVectorOnDevice(bool* dVector);
		
		}
	}
}