#include <thread>
#include "ComputeScheduler.cuh"
#include "DeviceInfo.cuh"
#include "DeviceComputeScheduler.cuh"
#include "ConcurrentQueue.h"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			CComputeScheduler::CComputeScheduler(bool* universePlacement, int virtualWidth, int virtualHeight, int bufferSize) {
				Initialize(universePlacement, virtualWidth, virtualHeight, bufferSize);
			}

			void CComputeScheduler::Initialize(
				bool* &universePlacement, const int &virtualWidth, const int &virtualHeight, const int &bufferSize) {

				this->virtualWidth = virtualWidth;
				this->virtualHeight = virtualHeight;
				realLength = virtualWidth * virtualHeight;

				InitializeDevice(universePlacement);

				precalculatedBuffer = new ConcurrentQueue<bool*>();
				precalculatedBufferLimit = bufferSize;

				state = Stopped;
			}

			void CComputeScheduler::InitializeDevice(bool* &universePlacement) {

				DeviceInfo* deviceInfo = new DeviceInfo();
				threadsPerBlock = deviceInfo->GetThreadsPerBlock(deviceInfo->GetMaxComputeCapabilityDeviceId());
				alignedBlocksCount = deviceInfo->GetAlignedBlocksCount(threadsPerBlock, virtualWidth * virtualHeight);
				alignedLength = alignedBlocksCount * threadsPerBlock;
				delete deviceInfo;

				CUDA::AllocateVectorOnDevice(dVector, alignedBlocksCount, threadsPerBlock);
				CUDA::AllocateVectorOnDevice(dBuffer, alignedBlocksCount, threadsPerBlock);

				cudaMemcpy(dVector, universePlacement, realLength * sizeof(bool), cudaMemcpyHostToDevice);
				readyToComputing = true;
			}

			void CComputeScheduler::Start() {
				if (state == Started) {
					Stop();
				}
				std::thread thr(&CComputeScheduler::StartProcess, this);
				thr.detach();
			}

			void CComputeScheduler::Stop() {
				state = WillBeStopped;
				while (state != Stopped) {
					_sleep(8);
				}
			}

			void CComputeScheduler::StartProcess() {

				state = Started;
				while (state == Started) {
					if (precalculatedBuffer->Count() < precalculatedBufferLimit) {

						bool* hResult = nullptr;
						DeviceGetNextGeneration(hResult);
						precalculatedBuffer->Enqueue(hResult);
					}
					else {
						_sleep(1);
					}
				}
				state = Stopped;
			}

			void CComputeScheduler::DeviceGetNextGeneration(bool* &hResult) {
				if (readyToComputing) {

					CUDA::Next << <alignedBlocksCount, threadsPerBlock >> >(dVector, dBuffer, virtualWidth, virtualHeight);
					CUDA::SwapValues << <alignedBlocksCount, threadsPerBlock >> >(dVector, dBuffer);
					CUDA::InitializeVectorOnDevice << <alignedBlocksCount, threadsPerBlock >> >(dBuffer, alignedLength);

					hResult == nullptr ? hResult = new bool[realLength] : NULL;
					cudaMemcpy(hResult, dVector, realLength * sizeof(bool), cudaMemcpyDeviceToHost);
				}
			}

			void CComputeScheduler::GetNextGeneration(bool* &universePlacement) {

				bool* localResult = nullptr;

				while (localResult == nullptr) {
					if (precalculatedBuffer->TryDequeue(localResult)) {
						universePlacement != nullptr ? delete universePlacement : NULL;
						universePlacement = localResult;
					}
					else {
						_sleep(1);
					}
				}
			}

			bool* CComputeScheduler::GetNextGeneration() {

				bool* localResult = nullptr;

				while (localResult == nullptr) {
					if (precalculatedBuffer->TryDequeue(localResult)) {
						return localResult;
					}
					else {
						_sleep(1);
					}
				}
			}

			CComputeScheduler::~CComputeScheduler() {
				Destroy();
			}

			void CComputeScheduler::Destroy() {

				if (state == Started) {
					Stop();
				}
				while (precalculatedBuffer->Count() > 0) {
					bool* tmp = nullptr;
					while (tmp == nullptr) {
						precalculatedBuffer->TryDequeue(tmp);
						tmp != nullptr ? delete tmp : NULL;
						tmp = nullptr;
					}
				}
				delete precalculatedBuffer;
				precalculatedBufferLimit = 0;

				CUDA::FreeVectorOnDevice(dVector);
				CUDA::FreeVectorOnDevice(dBuffer);
			}
		}
	}
}