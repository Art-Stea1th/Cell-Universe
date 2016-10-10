#pragma once


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			template<typename T> class ConcurrentQueue;

			class CComputeScheduler {

			private:

				enum FSMState { Started, WillBeStopped, Stopped };

				FSMState state;

				bool* dVector;
				bool* dBuffer;

				int virtualWidth, virtualHeight, realLength;
				int alignedBlocksCount, threadsPerBlock, alignedLength;

				bool readyToComputing = false;

				ConcurrentQueue<bool*>* precalculatedBuffer;
				int precalculatedBufferLimit;
				
				void Initialize(bool* &universePlacement, const int &virtualWidth, const int &virtualHeight, const int &bufferSize);
				void InitializeDevice(bool* &universePlacement);
				void StartProcess();
				void DeviceGetNextGeneration(bool* &hResult);
				void Destroy();

			public:

				void Start();
				void Stop();
				void GetNextGeneration(bool* &universePlacement);
				bool* GetNextGeneration();

				CComputeScheduler(bool* universePlacement, int virtualWidth, int virtualHeight, int bufferSize = 3);
				~CComputeScheduler();
			};
		}
	}
}