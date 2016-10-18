#include "DeviceComputeScheduler.cuh"
#include "DeviceInfo.cuh"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CUDA {

				DV int CycleFixIndex(int index, int vectorLength) {

					index %= vectorLength;

					if (index < 0) { return vectorLength + index; }
					if (index >= vectorLength) { return index - vectorLength; }

					return index;
				}

				DV int GetIndex1D(int x, int y, int width) {
					return y * width + x;
				}

				DV int CountNeighbourns(bool* dVector, int &index, int &width, int &height) {

					int x = index % width;
					int y = index / width;

					int count = 0;

					/*for (int offsetX = -1; offsetX <= 1; ++offsetX) {
						for (int offsetY = -1; offsetY <= 1; ++offsetY) {

							if (offsetX == 0 && offsetY == 0)
								continue;

							int nX = CycleFixIndex(x + offsetX, width);
							int nY = CycleFixIndex(y + offsetY, height);

							if (dVector[nY * width + nX])
								++count;
						}
					}*/

					if (dVector[GetIndex1D(CycleFixIndex(x - 1, width), CycleFixIndex(y - 1, height), width)]) ++count; // lt
					if (dVector[GetIndex1D(CycleFixIndex(x - 1, width), y, width)]) ++count;                            // lc
					if (dVector[GetIndex1D(CycleFixIndex(x - 1, width), CycleFixIndex(y + 1, height), width)]) ++count; // lb

					if (dVector[GetIndex1D(x, CycleFixIndex(y - 1, height), width)]) ++count; // ct
					if (dVector[GetIndex1D(x, CycleFixIndex(y + 1, height), width)]) ++count; // cb

					if (dVector[GetIndex1D(CycleFixIndex(x + 1, width), CycleFixIndex(y - 1, height), width)]) ++count; // rt
					if (dVector[GetIndex1D(CycleFixIndex(x + 1, width), y, width)]) ++count;                            // rc
					if (dVector[GetIndex1D(CycleFixIndex(x + 1, width), CycleFixIndex(y + 1, height), width)]) ++count; // rb

					return count;
				}

				GL void Next(bool* dVector, bool* dBuffer, int width, int height) {

					int length = width * height;
					int index = blockIdx.x * blockDim.x + threadIdx.x;

					if (index >= length) { return; }

					int aliveNeighboursCount = CountNeighbourns(dVector, index, width, height);

					if ((aliveNeighboursCount == 2 || aliveNeighboursCount == 3) && dVector[index]) {
						dBuffer[index] = true;
					}
					if (aliveNeighboursCount == 3 && !dVector[index]) {
						dBuffer[index] = true;
					}
				}

				GL void SwapValues(bool* dVector, bool* dBuffer) {

					int index = blockIdx.x * blockDim.x + threadIdx.x;

					bool tmp = dBuffer[index];
					dBuffer[index] = dVector[index];
					dVector[index] = tmp;
				}
			}
		}
	}
}