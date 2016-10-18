#include <random>
#include "Universe.h"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CPP {

				int CUniverse::countUniverses = 0;

				CUniverse::CUniverse(int width, int height) {
					this->virtualWidth = width;
					this->virtualHeight = height;
					universePlacement = new bool[width * height];
					FillRandom();
					scheduler = new CUDA::CComputeScheduler(universePlacement, virtualWidth, virtualHeight, 8);
					scheduler->Start();
					++countUniverses;
				}

				void CUniverse::FillRandom() {
					std::srand(countUniverses);
					int index = virtualWidth * virtualHeight;
					while (index--) {
						universePlacement[index] = std::rand() % 2 != 0;
					}
				}

				bool* CUniverse::GetNext() {
					scheduler->GetNextGeneration(universePlacement);
					return universePlacement;
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
}