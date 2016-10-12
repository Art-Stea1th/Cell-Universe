#pragma once
#include "IFSMCompatible.h"

namespace CellUniverse {
	namespace Models {
		namespace Native {

			namespace CPP { template<typename T> class ConcurrentQueue; }

			namespace CUDA {
				
				class DeviceInfo;

				class CComputeScheduler : public IFSMCompatible {

				private:

					static DeviceInfo* devInfo;

					enum State { Started, WillBeStopped, Stopped };

					State state;

					bool* dVector;
					bool* dBuffer;

					int virtualWidth, virtualHeight, realLength;
					int alignedBlocksCount, threadsPerBlock, alignedLength;

					bool readyToComputing = false;

					CPP::ConcurrentQueue<bool*>* precalculatedBuffer;
					unsigned precalculatedBufferLimit;

					void OnStarted() override;

					void Initialize(bool* &universe, const int &width, const int &height, const int &bufferSize);
					void InitializeDevice(bool* &universe);
					void StartProcess();
					void DeviceGetNextGeneration(bool* &hResult);
					void Destroy();

				public:

					void Start();
					void Stop();
					void GetNextGeneration(bool* &universePlacement);

					CComputeScheduler(bool* universe, int width, int height, int buffSize = 3);
					~CComputeScheduler();
				};
			}
		}
	}
}