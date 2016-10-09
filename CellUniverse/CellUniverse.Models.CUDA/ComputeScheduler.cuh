#pragma once
#include <thread>
#include "ConcurrentQueue.h"
#include "DefinesCUDA.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			class CComputeScheduler {

			private:

				bool* universePlacement;

				int precalculatedBufferCount;
				int precalculatedBufferLimit;

				bool thisWillBeDestroyed;

				ConcurrentQueue<bool*>* precalculatedBuffer;

				int deviceIndex;
				cudaDeviceProp* deviceProp;				

				void Initialize(bool* &universePlacement, const int &bufferSize);
				void Destroy();

			public:

				void Start();
				bool* GetNextGeneration();

				CComputeScheduler(bool* universePlacement, int bufferSize = 3);
				~CComputeScheduler();
			};
		}
	}
}