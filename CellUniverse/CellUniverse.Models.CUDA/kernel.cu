// GTX 550 Ti
// streaming multiprocessors    = 4
// scalar processors per/SM     = 48 (CUDA-cores, parallel warps)
// Total CUDA-cores (4 * 48)    = 192
// Warp size (threads per warp) = 32
// Max threads per block        = 1024
// Max threads per SM           = 1536

#include "DeviceInfo.cuh"
#include "MemoryManager.cuh"
#include "Universe.cuh"
#include "ComputeScheduler.cuh"
#include "ConcurrentQueue.h"
#include <iostream>

using namespace CellUniverse::Models::CUDA;

int main() {

	int maxCcDevId = GetMaxComputeCapabilityDeviceId();
	cudaSetDevice(maxCcDevId);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, maxCcDevId);

	int width = 1920, height = 1080, blocksCount = width * height;

	int threadsPerBlock = GetThreadsPerBlock(deviceProp);
	int alignedBlocksCount = GetAlignedBlocksCount(threadsPerBlock, blocksCount);

	bool* hUniverse;
	bool* dUniverse;

	AllocateVectorOnHost(hUniverse, blocksCount);
	AllocateVectorOnDevice(dUniverse, alignedBlocksCount, threadsPerBlock);

	FreeVectorOnHost(hUniverse);
	FreeVectorOnDevice(dUniverse);
	
	unsigned char i = 0;

	while (--i) {		

		CUniverse* universe = new CUniverse(width, height);
		delete universe;

		//_sleep(8);
	}

	ConcurrentQueue<bool*>* q = new ConcurrentQueue<bool*>();

	for (int i = 0; i < 1000; ++i) {
		bool* tmp = new bool(i % 2);
		q->Enqueue(tmp);
	}

	while (q->Count() > 0) {
		bool* tmp = nullptr;
		if (q->TryDequeue(tmp)) {
			std::cout << *tmp << ' ';
			delete tmp;
		}
	}

	std::cout << '\n' << alignedBlocksCount << '\n' << threadsPerBlock << '\n';
	std::cout << "ok\n";
	system("pause"); return 0;
}