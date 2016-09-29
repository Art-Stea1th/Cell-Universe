#pragma once


namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			class __declspec(dllexport) CTheGameOfLife {

			private:

				bool** next_result;

			public:

				bool** GetNextGeneration();

				CTheGameOfLife(const int &width, const int &height);
				~CTheGameOfLife();
			};
		}
	}
}