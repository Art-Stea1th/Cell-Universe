#include <random>
#include "Universe.cuh"

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			int CUniverse::countUniverses = 0;

			CUniverse::CUniverse(int width, int height) {
				this->virtualWidth = width;
				this->virtualHeight = height;
				universePlacement = new bool[width * height];
				FillRandom();
				scheduler = new CComputeScheduler(universePlacement, virtualWidth, virtualHeight, 3);	
				scheduler->Start();
				++countUniverses;
			}

			void CUniverse::FillRandom() {
				std::srand(countUniverses);
				int index = virtualWidth * virtualHeight;
				while (index--) {
					universePlacement[index] = static_cast<bool>(std::rand() % 2);
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