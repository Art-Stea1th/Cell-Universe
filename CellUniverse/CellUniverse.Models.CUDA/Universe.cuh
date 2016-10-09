#pragma once

#include "Defines.cuh"
#include "ComputeScheduler.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			class EXPORTED CUniverse {

			private:

				int virtualWidth, virtualHeight;
				bool* universePlacement;

				CComputeScheduler* scheduler;

			public:

				CUniverse(int width, int height);
				~CUniverse();
			};
		}
	}
}