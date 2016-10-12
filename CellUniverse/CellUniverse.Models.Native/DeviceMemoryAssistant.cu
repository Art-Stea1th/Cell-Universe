#include "DeviceMemoryAssistant.cuh"


namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CUDA {

				HT void AllocateVectorOnDevice(bool* &dVector, int alignedBlocksCount, int threadsPerBlock) {
					cudaMalloc((void**)&dVector, alignedBlocksCount * threadsPerBlock * sizeof(bool));
					InitializeVectorOnDevice<<<alignedBlocksCount, threadsPerBlock>>>(dVector, alignedBlocksCount * threadsPerBlock);
				}

				GL void InitializeVectorOnDevice(bool* dVector, int alignedLength) {
					const int index = blockDim.x * blockIdx.x + threadIdx.x;

					if (index < alignedLength) {
						dVector[index] = false;
					}
				}

				HT void FreeVectorOnDevice(bool* dVector) {
					cudaFree(&dVector);
					dVector = nullptr;
				}
			}
		}
	}
}