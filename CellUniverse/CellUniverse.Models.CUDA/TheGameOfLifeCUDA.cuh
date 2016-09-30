#pragma once
#include <cstdint>
#include <ctime>
#include <random>

#include <CUDA_runtime.h>
#include <device_launch_parameters.h>

#define GLOBAL __global__
#define DEVICE __device__
#define HOST __host__

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			void InitializeDevice(bool** &matrix, int width, int height);
			void ConfigureDevice(int width, int height);
			void AllocateLinearMatrixOnDevice(bool* &matrix, int width, int height);
			GLOBAL void InitializeLinearMatrixOnDevice(bool* matrix);

			void CalculateNextGeneration(bool** &matrix);
			GLOBAL void DoNextStep(bool* dev_matrix, bool* dev_buffer);
			DEVICE int CountNeighbours(bool* &dev_matrix, int width, int height, int posX, int posY);
			GLOBAL void SwapValues(bool* dev_matrix, bool* dev_buffer);

			void FreeDevice();
			void DestroyLinearMatrixOnDevice(bool* matrix);
		}
	}
}