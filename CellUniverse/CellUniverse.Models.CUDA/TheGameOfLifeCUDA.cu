#include "TheGameOfLifeCUDA.cuh"


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			int _width;
			int _height;
			size_t _size;

			dim3 _blocks_grid;

			bool* _dev_matrix;
			bool* _dev_buffer;


			void InitializeDevice(bool** &matrix, int width, int height) {

				ConfigureDevice(width, height);

				AllocateLinearMatrixOnDevice(_dev_matrix, width, height);
				AllocateLinearMatrixOnDevice(_dev_buffer, width, height);

				cudaMemcpy(_dev_matrix, matrix[0], _size, cudaMemcpyHostToDevice);
			}

			void ConfigureDevice(int width, int height) {
				_width = width; _height = height;
				_size = width * height * sizeof(bool);
				cudaSetDevice(0);
				_blocks_grid = dim3(_width, _height);				
			}

			void AllocateLinearMatrixOnDevice(bool* &matrix, int width, int height) {

				cudaMalloc((void**)&matrix, _size);
				InitializeLinearMatrixOnDevice<<<_blocks_grid, 1>>>(matrix);
			}

			GLOBAL void InitializeLinearMatrixOnDevice(bool* matrix) {

				int width = gridDim.x;
				int x = blockIdx.x;
				int y = blockIdx.y;

				int index1D = width * y + x;

				matrix[index1D] = false;
			}

			void CalculateNextGeneration(bool** &matrix) {
				DoNextStep<<<_blocks_grid, 1>>>(_dev_matrix, _dev_buffer);
				SwapValues<<<_blocks_grid, 1>>>(_dev_matrix, _dev_buffer);
				InitializeLinearMatrixOnDevice<<<_blocks_grid, 1>>>(_dev_buffer);
				cudaMemcpy(matrix[0], _dev_matrix, _size, cudaMemcpyDeviceToHost);
			}

			GLOBAL void DoNextStep(bool* dev_matrix, bool* dev_buffer) {

				int width = gridDim.x;
				int height = gridDim.y;
				int x = blockIdx.x;
				int y = blockIdx.y;

				int index1D = width * y + x;

				int neighboursCount = CountNeighbours(dev_matrix, width, height, x, y);

				if ((neighboursCount == 2 || neighboursCount == 3) && dev_matrix[index1D]) {
					dev_buffer[index1D] = true;
				}
				if ((neighboursCount < 2 || neighboursCount > 3) && dev_matrix[index1D]) {
					dev_buffer[index1D] = false;
				}
				if (neighboursCount == 3 && !dev_matrix[index1D]) {
					dev_buffer[index1D] = true;
				}
			}			

			DEVICE int CountNeighbours(bool* &dev_matrix, int width, int height, int posX, int posY) {

				int counter = 0;

				int startX = posX - 1;
				int endX = posX + 1;
				int startY = posY - 1;
				int endY = posY + 1;

				for (int y = startY; y <= endY; ++y) {
					for (int x = startX; x <= endX; ++x) {
						if (x == posX && y == posY)
							continue;

						int px = x, py = y;

						if (px == -1) px = width - 1;
						else if (px == width) px = 0;

						if (py == -1) py = height - 1;
						else if (py == height) py = 0;

						if (dev_matrix[width * py + px])
							counter++;
					}
				}
				return counter;
			}

			GLOBAL void SwapValues(bool* dev_matrix, bool* dev_buffer) {

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