#pragma once
#include <cstdint>

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			using namespace std;
			typedef uint_fast8_t byte;

			class __declspec(dllexport) CTheGameOfLife {

			private:

				int width;
				int height;

				bool** generation;
				bool** buffer;

				bool* dev_generation;
				bool* dev_buffer;

				void AllocateLinearMatrix(bool** &matrix, const int &width, const int &height);
				void InitializeLinearMatrix(bool** &matrix, const int &width, const int &height);
				void FillRandomLinearMatrix(bool** &matrix, const int &width, const int &height);
				void DestroyLinearMatrix(bool** &matrix);

				void CalculateNextGeneration();
				byte CountNeighbours(bool** &generation, const int &posX, const int &posY);

			public:

				bool** GetNextGeneration();
				bool TryGetNextGeneration(int &posX, int &posY, bool &value);

				CTheGameOfLife(const int &width, const int &height);
				~CTheGameOfLife();
			};
		}
	}
}