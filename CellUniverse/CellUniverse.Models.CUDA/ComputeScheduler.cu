#include "ComputeScheduler.cuh"
#include "DeviceInfo.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {
			
			CComputeScheduler::CComputeScheduler(bool* universePlacement, int bufferSize) {
				Initialize(universePlacement, bufferSize);
			}

			void CComputeScheduler::Initialize(bool* &universePlacement, const int &bufferSize) {

				this->universePlacement = universePlacement;

				precalculatedBufferCount = 0;
				precalculatedBufferLimit = bufferSize;

				thisWillBeDestroyed = false;

				deviceIndex = GetMaxComputeCapabilityDeviceId();
				deviceProp = new cudaDeviceProp();
				cudaGetDeviceProperties(deviceProp, deviceIndex);				
			}

			void CComputeScheduler::Start() {

				while (!thisWillBeDestroyed) {
					if (precalculatedBufferCount < precalculatedBufferLimit) {

						bool* localBuffer; // = cuda result

						precalculatedBuffer->Enqueue(localBuffer);
						//Interlocked.Increment(ref precalculatedBufferCount);
					}
					else {
						_sleep(1);
					}
				}
			}

			bool* CComputeScheduler::GetNextGeneration() {
				return nullptr;
			}

			CComputeScheduler::~CComputeScheduler() {
				Destroy();
			}

			void CComputeScheduler::Destroy() {

				universePlacement = nullptr;

				precalculatedBufferCount = 0;
				precalculatedBufferLimit = 0;

				thisWillBeDestroyed = true;

				deviceIndex = 0;
				delete deviceProp;
			}
		}
	}
}