#include "Universe.cuh"


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			CUniverse::CUniverse(int width, int height) {
				this->virtualWidth = width;
				this->virtualHeight = height;
				universePlacement = new bool[width * height];
				scheduler = new CComputeScheduler(universePlacement, 3);
			}

			CUniverse::~CUniverse() {
				delete scheduler;
				delete universePlacement;
				universePlacement = nullptr;
				virtualWidth = virtualHeight = 0;
			}
		}
	}
}