#pragma once

#include "Defines.h"
#include "ComputeScheduler.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			class EXPORTED CUniverse {

			private:

				static int countUniverses;

				bool* universePlacement;
				int virtualWidth, virtualHeight;				

				CComputeScheduler* scheduler;

				void FillRandom();

			public:

				bool* GetNext();

				CUniverse(int width, int height);
				~CUniverse();
			};
		}
	}
}