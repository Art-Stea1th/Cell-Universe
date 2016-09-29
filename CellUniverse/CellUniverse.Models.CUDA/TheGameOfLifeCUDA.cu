#include "TheGameOfLifeCUDA.cuh"


namespace CellUniverse {
	namespace Models {
		namespace CUDA {			

			// --- SHARED ------------

			int _width;
			int _height;

			// --- DEVICE ------------

			dim3 _blocks_grid;

			bool* _dev_generation;
			bool* _dev_buffer;


			// --- SHARED ----------------------------------------------------------------------------------------------

			void Initialize(bool** &matrix, const int &width, const int &height) {

				_width = width; _height = height;				

				AllocateLinearMatrixOnHost(matrix, width, height);				
				FillRandomLinearMatrixOnHost(matrix, width, height);				

				Configure(width, height);

				AllocateLinearMatrixOnDevice(_dev_generation, width, height);
				AllocateLinearMatrixOnDevice(_dev_buffer, width, height);

				cudaMemcpy(_dev_generation, matrix[0], width * height * sizeof(bool), cudaMemcpyHostToDevice);
			}
			
			void CalculateNextGeneration(bool** &matrix) {
				DoNextStep<<<_blocks_grid, 1>>>(_dev_generation, _dev_buffer);
				SwapValues<<<_blocks_grid, 1>>>(_dev_generation, _dev_buffer);
				InitializeLinearMatrixOnDevice<<<_blocks_grid, 1>>>(_dev_buffer);
				cudaMemcpy(matrix[0], _dev_generation, _width * _height * sizeof(bool), cudaMemcpyDeviceToHost);
			}

			GLOBAL void DoNextStep(bool* dev_matrix, bool* dev_buffer) {

				int width = gridDim.x;
				int height = gridDim.y;
				int col = blockIdx.x;
				int row = blockIdx.y;

				int index1D = width * row + col;

				int neighboursCount = CountNeighbours(dev_matrix, width, height, col, row);

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

			GLOBAL void SwapValues(bool* dev_matrix, bool* dev_buffer) {

				int width = gridDim.x;
				int col = blockIdx.x;
				int row = blockIdx.y;

				int index1D = width * row + col;

				bool tmp = dev_matrix[index1D];
				dev_matrix[index1D] = dev_buffer[index1D];
				dev_buffer[index1D] = tmp;
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

			void Destroy(bool** &matrix) {
				
				DestroyLinearMatrixOnDevice(_dev_buffer);
				DestroyLinearMatrixOnDevice(_dev_generation);

				DestroyLinearMatrixOnHost(matrix);
			}

			// --- HOST ------------------------------------------------------------------------------------------------						

			void AllocateLinearMatrixOnHost(bool** &matrix, const int &width, const int &height) {

				matrix = new bool*[height];
				matrix[0] = new bool[height * width];

				for (int i(1); i < height; ++i) {
					matrix[i] = matrix[i - 1] + width;
				}
				InitializeLinearMatrixOnHost(matrix, width, height);
			}

			void InitializeLinearMatrixOnHost(bool** &matrix, const int &width, const int &height) {

				int index = width * height;

				while (index--) {
					matrix[0][index] = false;
				}
			}

			void FillRandomLinearMatrixOnHost(bool** &matrix, const int &width, const int &height) {

				int index = width * height; std::random_device random;

				while (index--) {
					matrix[0][index] = static_cast<bool>(random() % 2);
				}
			}

			void DestroyLinearMatrixOnHost(bool** &matrix) {
				delete[] matrix[0];
				matrix[0] = nullptr;
				delete[] matrix;
				matrix = nullptr;
			}

			// --- DEVICE ----------------------------------------------------------------------------------------------			

			void Configure(const int &width, const int &height) {
				_blocks_grid = dim3(width, height);
				cudaSetDevice(0);
			}

			void AllocateLinearMatrixOnDevice(bool* &matrix, const int &width, const int &height) {

				cudaMalloc((void**)&matrix, width * height * sizeof(bool));				
				InitializeLinearMatrixOnDevice<<<_blocks_grid, 1>>>(matrix);
			}

			GLOBAL void InitializeLinearMatrixOnDevice(bool* matrix) {

				int width = gridDim.x;
				int col = blockIdx.x;
				int row = blockIdx.y;

				int index1D = width * row + col;
				
				matrix[index1D] = false;
			}

			void DestroyLinearMatrixOnDevice(bool* &matrix) {
				cudaFree(matrix);
			}
		}
	}
}