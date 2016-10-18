#pragma once
#include "DefinesCUDA.cuh"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CUDA {

				GL void Next(bool* dVector, bool* dBuffer, int width, int height);
				GL void SwapValues(bool* dVector, bool* dBuffer);

			}
		}
	}
}