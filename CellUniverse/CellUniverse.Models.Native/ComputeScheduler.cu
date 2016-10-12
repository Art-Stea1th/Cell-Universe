#include "Threading.h"
#include "DeviceInfo.cuh"
#include "DeviceMemoryAssistant.cuh"
#include "DeviceComputeScheduler.cuh"
#include "ConcurrentQueue.h"
#include "ComputeScheduler.cuh"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CUDA {

				DeviceInfo* CComputeScheduler::devInfo = new DeviceInfo();

				CComputeScheduler::CComputeScheduler(bool* universe, int width, int height, int buffSize) {					
					Initialize(universe, width, height, buffSize);
					InitializeDevice(universe);
				}

				void CComputeScheduler::Initialize(bool* &universe, const int &width, const int &height, const int &buffSize) {

					virtualWidth  = width;
					virtualHeight = height;
					realLength = width * height;
					
					precalculatedBuffer = new CPP::ConcurrentQueue<bool*>();
					precalculatedBufferLimit = buffSize;

					state = Stopped;
				}

				void CComputeScheduler::InitializeDevice(bool* &universe) {
					
					threadsPerBlock = devInfo->GetThreadsPerBlock(devInfo->GetMaxComputeCapabilityDeviceId());
					alignedBlocksCount = devInfo->GetAlignedBlocksCount(threadsPerBlock, realLength);
					alignedLength = alignedBlocksCount * threadsPerBlock;

					CUDA::AllocateVectorOnDevice(dVector, alignedBlocksCount, threadsPerBlock);
					CUDA::AllocateVectorOnDevice(dBuffer, alignedBlocksCount, threadsPerBlock);

					cudaMemcpy(dVector, universe, realLength * sizeof(bool), cudaMemcpyHostToDevice);
					readyToComputing = true;
				}

				void CComputeScheduler::OnStarted() {

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
					while (state != Stopped) { SLEEP(1); }
				}

				void CComputeScheduler::StartProcess() {

					state = Started;
					while (state == Started) {
						if (precalculatedBuffer->Count() < precalculatedBufferLimit) {

							bool* hResult = nullptr;
							DeviceGetNextGeneration(hResult);
							precalculatedBuffer->Enqueue(hResult);
						}
						else { SLEEP(1); }
					}
					state = Stopped;					
				}

				void CComputeScheduler::DeviceGetNextGeneration(bool* &hResult) {
					if (readyToComputing) {

						CUDA::Next<<<alignedBlocksCount, threadsPerBlock>>>(dVector, dBuffer, virtualWidth, virtualHeight);
						CUDA::SwapValues<<<alignedBlocksCount, threadsPerBlock>>>(dVector, dBuffer);
						CUDA::InitializeVectorOnDevice<<<alignedBlocksCount, threadsPerBlock>>>(dBuffer, alignedLength);

						hResult == nullptr ? hResult = new bool[realLength] : NULL;
						cudaMemcpy(hResult, dVector, realLength * sizeof(bool), cudaMemcpyDeviceToHost);
					}
				}

				void CComputeScheduler::GetNextGeneration(bool* &universe) {

					bool* localResult = nullptr;

					while (localResult == nullptr) {
						if (precalculatedBuffer->TryDequeue(localResult)) {
							universe != nullptr ? delete universe : NULL;
							universe = localResult;
						}
						else { SLEEP(1); }
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

					if (devInfo != nullptr) {
						delete devInfo;
					}

					CUDA::FreeVectorOnDevice(dVector);
					CUDA::FreeVectorOnDevice(dBuffer);
				}
			}
		}
	}
}