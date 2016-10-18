#pragma once

#include "Defines.h"
#include "ComputeScheduler.cuh"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CPP {

				class exported CUniverse {

				private:

					static int countUniverses;

					bool* universePlacement;
					int virtualWidth, virtualHeight;

					CUDA::CComputeScheduler* scheduler;

					void FillRandom();

				public:

					bool* GetNext();

					CUniverse(int width, int height);
					~CUniverse();
				};
			}
		}
	}
}