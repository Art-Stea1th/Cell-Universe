#include "DeviceInfo.cuh"
#include "Defines.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			HT int GetMaxComputeCapabilityDeviceId() {

				int deviceCount = 0;
				cudaGetDeviceCount(&deviceCount);

				cudaDeviceProp* deviceProps = nullptr;

				if (deviceCount > 0) {

					deviceProps = new cudaDeviceProp[deviceCount];

					int maxComputeCapabilityDeviceId = 0;
					int localMajor = 0, localMinor = 0;

					for (int i = 0; i < deviceCount; ++i) {
						cudaGetDeviceProperties(&deviceProps[i], i);
						if (deviceProps[i].major > localMajor) {
							maxComputeCapabilityDeviceId = i;
							continue;
						}
						if (deviceProps[i].major == localMajor) {
							if (deviceProps[i].minor > localMinor) {
								maxComputeCapabilityDeviceId = i;
							}
						}
					}
					deviceProps != nullptr ? delete deviceProps : NOTHING;
					return maxComputeCapabilityDeviceId;
				}
				else {
					return -1;
				}
			}

			HT int GetThreadsPerBlock(const cudaDeviceProp &deviceProp) {

				int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
				int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

				int threadsPerBlock = maxThreadsPerSM;

				while (threadsPerBlock > maxThreadsPerBlock) {
					threadsPerBlock >>= 1;
				}
				return threadsPerBlock;
			}

			HT int GetAlignedBlocksCount(int threadsPerBlock, int vectorLength) {
				return vectorLength % threadsPerBlock != 0
					? vectorLength / threadsPerBlock + 1
					: vectorLength / threadsPerBlock;
			}
		}
	}
}