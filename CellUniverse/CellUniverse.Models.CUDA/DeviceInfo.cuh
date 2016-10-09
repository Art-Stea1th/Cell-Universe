#pragma once

#include "DefinesCUDA.cuh"


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			HT int GetMaxComputeCapabilityDeviceId();
			HT int GetThreadsPerBlock(const cudaDeviceProp &deviceProp);
			HT int GetAlignedBlocksCount(int threadsPerBlock, int vectorLength);

		}
	}
}