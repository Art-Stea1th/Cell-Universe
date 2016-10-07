#pragma once
#include <cstdint>
#include <ctime>
#include <random>

#include <CUDA_runtime.h>
#include <device_launch_parameters.h>


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			void InitializeDevice(bool* &matrix, int width, int height);
			void ConfigureDevice(int width, int height);
			void AllocateLinearMatrixOnDevice(bool* &matrix, int width, int height);
			__global__ void InitializeLinearMatrixOnDevice(bool* matrix);

			void CalculateNextGeneration(bool* &matrix);
			__global__ void DoNextStep(bool* dev_matrix, bool* dev_buffer);
			__device__ int CycleFixIndex(int index, int vectorLength);
			__global__ void SwapValues(bool* dev_matrix, bool* dev_buffer);

			void FreeDevice();
			void DestroyLinearMatrixOnDevice(bool* matrix);
		}
	}
}