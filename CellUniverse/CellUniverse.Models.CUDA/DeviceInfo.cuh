#pragma once


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			class DeviceInfo {
			
			private:

				int deviceCount;
				cudaDeviceProp* deviceProps;

				void Initialize();

			public:

				int GetMaxComputeCapabilityDeviceId();
				int GetThreadsPerBlock(int deviceIndex);
				int GetAlignedBlocksCount(int threadsPerBlock, int vectorLength);

				DeviceInfo();
				~DeviceInfo();
			};
		}
	}
}