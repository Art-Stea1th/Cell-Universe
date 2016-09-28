#pragma once
#include <stack>
#include "CellInfo.h"

namespace CellUniverse {
	namespace Models {
		namespace Unmanaged {

			using namespace std;
			typedef uint_fast8_t byte;

			class __declspec(dllexport) CTheGameOfLifeNative {

			private:

				int width;
				int height;

				bool** generation;
				bool** buffer2d;
				
				void AllocateMatrix(bool** &matrix, const int &width, const int &height);
				void InitializeMatrix(bool** &matrix, const int &width, const int &height);
				void FillRandomMatrix(bool** &matrix, const int &width, const int &height);
				void DestroyMatrix(bool** &matrix);

				void CalculateNextGeneration();
				byte CountNeighbours(bool** &generation, const int &posX, const int &posY);				

			public:

				bool** GetNextGeneration();
				bool TryGetNextGeneration(int &posX, int &posY, bool &value);

				CTheGameOfLifeNative(const int &width, const int &height);
				~CTheGameOfLifeNative();
			};
		}
	}	
}