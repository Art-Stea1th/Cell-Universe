#pragma once
#include <cstdint>
#include <ctime>
#include <random>

#include <CUDA_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLOBAL __global__
#define DEVICE __device__
#define HOST __host__

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			// --- SHARED ---

			void Initialize(bool** &matrix, const int &width, const int &height);
			void CalculateNextGeneration(bool** &matrix);
			void Destroy(bool** &matrix);

			// --- HOST ---

			void AllocateLinearMatrixOnHost(bool** &matrix, const int &width, const int &height);
			void InitializeLinearMatrixOnHost(bool** &matrix, const int &width, const int &height);
			void FillRandomLinearMatrixOnHost(bool** &matrix, const int &width, const int &height);
			void DestroyLinearMatrixOnHost(bool** &matrix);

			// --- DEVICE ---

			void Configure(const int &width, const int &height);

			void AllocateLinearMatrixOnDevice(bool* &matrix, const int &width, const int &height);
			GLOBAL void InitializeLinearMatrixOnDevice(bool* matrix);

			GLOBAL void DoNextStep(bool* dev_matrix, bool* dev_buffer);
			DEVICE int CountNeighbours(bool* &dev_matrix, int width, int height, int posX, int posY);
			GLOBAL void SwapValues(bool* dev_matrix, bool* dev_buffer);

			void DestroyLinearMatrixOnDevice(bool* &matrix);
		}
	}
}