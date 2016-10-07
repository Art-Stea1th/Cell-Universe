#include "Universe.cuh"


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			int _width;
			int _height;
			size_t _size;

			dim3 _blocks_grid;
			dim3 _neighbourns_grid;

			bool* _dev_matrix;
			bool* _dev_buffer;


			void InitializeDevice(bool* &matrix, int width, int height) {

				ConfigureDevice(width, height);

				AllocateLinearMatrixOnDevice(_dev_matrix, width, height);
				AllocateLinearMatrixOnDevice(_dev_buffer, width, height);

				cudaMemcpy(_dev_matrix, matrix, _size, cudaMemcpyHostToDevice);
			}

			void ConfigureDevice(int width, int height) {
				_width = width; _height = height;
				_size = width * height * sizeof(bool);
				cudaSetDevice(0);
				_blocks_grid = dim3(_width, _height);
				_neighbourns_grid = dim3(3, 3);
			}

			void AllocateLinearMatrixOnDevice(bool* &matrix, int width, int height) {

				cudaMalloc((void**)&matrix, _size);
				InitializeLinearMatrixOnDevice<<<_blocks_grid, 1>>>(matrix);
			}

			__global__ void InitializeLinearMatrixOnDevice(bool* matrix) {

				int width = gridDim.x;
				int x = blockIdx.x;
				int y = blockIdx.y;

				int index1D = width * y + x;

				matrix[index1D] = false;
			}

			void CalculateNextGeneration(bool* &matrix) {
				DoNextStep<<<_blocks_grid, dim3(3, 3)>>>(_dev_matrix, _dev_buffer);
				SwapValues<<<_blocks_grid, 1>>>(_dev_matrix, _dev_buffer);
				InitializeLinearMatrixOnDevice<<<_blocks_grid, 1>>>(_dev_buffer);
				cudaMemcpy(matrix, _dev_matrix, _size, cudaMemcpyDeviceToHost);
			}

			__global__ void DoNextStep(bool* dev_matrix, bool* dev_buffer) {

				__shared__ int aliveNeighboursCount;

				if (threadIdx.x == 0 && threadIdx.y == 0) {
					aliveNeighboursCount = 0;
				}
				__syncthreads();

				int fieldWidth = gridDim.x;
				int fieldHeight = gridDim.y;

				int cellX = blockIdx.x;
				int cellY = blockIdx.y;

				int neighbourOffsetX = threadIdx.x - 1; // -1, 0, +1
				int neighbourOffsetY = threadIdx.y - 1; // -1, 0, +1

				int neighbourX = CycleFixIndex(cellX + neighbourOffsetX, fieldWidth);
				int neighbourY = CycleFixIndex(cellY + neighbourOffsetY, fieldHeight);

				int cellLinearIndex = cellX + cellY * fieldWidth;
				int neighbourLinearIndex = neighbourX + neighbourY * fieldWidth;

				if (dev_matrix[neighbourLinearIndex] && cellLinearIndex != neighbourLinearIndex) {
					atomicAdd(&aliveNeighboursCount, 1);
				}
				__syncthreads();

				if (threadIdx.x == 0 && threadIdx.y == 0) {

					bool isAlive = dev_matrix[cellLinearIndex];

					if ((aliveNeighboursCount == 2 || aliveNeighboursCount == 3) && isAlive) {
						dev_buffer[cellLinearIndex] = true;
					}
					if (aliveNeighboursCount == 3 && !isAlive) {
						dev_buffer[cellLinearIndex] = true;
					}
				}
			}

			__device__ int CycleFixIndex(int index, int vectorLength) {

				index %= vectorLength;

				if (index < 0) {
					return vectorLength + index;
				}
				if (index >= vectorLength) {
					return index - vectorLength;
				}

				return index;
			}

			__global__ void SwapValues(bool* dev_matrix, bool* dev_buffer) {

				int width = gridDim.x;
				int x = blockIdx.x;
				int y = blockIdx.y;

				int index1D = width * y + x;

				bool tmp = dev_matrix[index1D];
				dev_matrix[index1D] = dev_buffer[index1D];
				dev_buffer[index1D] = tmp;				
			}

			void FreeDevice() {
				DestroyLinearMatrixOnDevice(_dev_buffer);
				DestroyLinearMatrixOnDevice(_dev_matrix);
			}

			void DestroyLinearMatrixOnDevice(bool* matrix) {
				cudaFree(&matrix);
				matrix = nullptr;
			}
		}
	}
}