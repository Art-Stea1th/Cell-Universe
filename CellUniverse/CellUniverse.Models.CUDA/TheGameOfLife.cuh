#pragma once


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			class __declspec(dllexport) CTheGameOfLife {

			private:

				bool* matrix;

				void AllocateLinearMatrix(const int &width, const int &height);
				void InitializeLinearMatrix(const int &width, const int &height);
				void FillRandomLinearMatrix(const int &width, const int &height);
				void DestroyLinearMatrix();

			public:

				bool* GetNextGeneration();

				CTheGameOfLife(const int &width, const int &height);
				~CTheGameOfLife();
			};
		}
	}
}