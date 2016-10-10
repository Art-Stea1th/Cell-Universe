#include "DefinesCUDA.cuh"
#include "DeviceInfo.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			DeviceInfo::DeviceInfo() {
				deviceCount = 0;
				deviceProps = nullptr;
				Initialize();
			}

			void DeviceInfo::Initialize() {
				cudaGetDeviceCount(&deviceCount);				
				if (deviceCount > 0) {
					deviceProps = new cudaDeviceProp[deviceCount];
					for (int i = 0; i < deviceCount; ++i) {
						cudaGetDeviceProperties(&deviceProps[i], i);
					}
				}
			}

			int DeviceInfo::GetMaxComputeCapabilityDeviceId() {

				if (deviceCount > 0) {

					int maxComputeCapabilityDeviceId = 0;
					int localMajor = 0, localMinor = 0;

					for (int i = 0; i < deviceCount; ++i) {
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
					return maxComputeCapabilityDeviceId;
				}
				else {
					return -1;
				}
			}					

			int DeviceInfo::GetThreadsPerBlock(int deviceIndex) {

				int maxThreadsPerSM = deviceProps[deviceIndex].maxThreadsPerMultiProcessor;
				int maxThreadsPerBlock = deviceProps[deviceIndex].maxThreadsPerBlock;

				int threadsPerBlock = maxThreadsPerSM;

				while (threadsPerBlock > maxThreadsPerBlock) {
					threadsPerBlock >>= 1;
				}
				return threadsPerBlock;
			}

			int DeviceInfo::GetAlignedBlocksCount(int threadsPerBlock, int vectorLength) {
				return threadsPerBlock == 0 ? 0 :
					(vectorLength % threadsPerBlock == 0
						? vectorLength / threadsPerBlock
						: vectorLength / threadsPerBlock + 1);				
			}

			DeviceInfo::~DeviceInfo() {
				deviceProps != nullptr ? delete deviceProps : NULL;
			}
		}
	}
}