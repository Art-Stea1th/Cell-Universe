#include "MemoryManager.cuh"

#ifndef MEMORY_MANAGER
#define MEMORY_MANAGER

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			template <typename T>
			HT void AllocateVectorOnHost(T* &hPtr, int vectorLength) {
				hPtr = new T[vectorLength];
				InitializeVectorOnHost(hPtr, vectorLength);
			}

			template <typename T>
			HT void InitializeVectorOnHost(T* &hPtr, int vectorLength) {
				while (vectorLength--) {
					hPtr[vectorLength] = static_cast<T>(0);
				}
			}

			template<typename T>
			HT void FreeVectorOnHost(T* hPtr) {
				delete hPtr;
				hPtr = nullptr;
			}


			template <typename T>
			HT void AllocateVectorOnDevice(T* &dPtr, int alignedBlocksCount, int threadsPerBlock) {
				cudaMalloc((void**)&dPtr, sizeof(T) * alignedBlocksCount * threadsPerBlock);
				InitializeVectorOnDevice<<<alignedBlocksCount, threadsPerBlock>>>(dPtr, alignedBlocksCount, threadsPerBlock);
			}

			template <typename T>
			GL void InitializeVectorOnDevice(T* &dPtr, int alignedBlocksCount, int threadsPerBlock) {

				const int vectorLength = alignedBlocksCount * threadsPerBlock;
				const int linearIndex = threadsPerBlock * blockIdx.x + threadIdx.x;

				if (linearIndex < vectorLength) {
					dPtr[linearIndex] = static_cast<T>(0);
				}
			}

			template<typename T>
			HT void FreeVectorOnDevice(T* dPtr) {
				cudaFree(&dPtr);
				dPtr = nullptr;
			}
		}
	}
}

#endif // !MEMORY_MANAGER