#pragma once
#define CLI_MODEL

namespace CellUniverse {
	namespace Models {

		namespace CUDA { class CUniverse; }

		namespace CLI {

			public ref class CTheGameOfLife {

			private:

				CUDA::CUniverse * _impl;

			public:

				bool* GetNextGeneration();

				void Destroy();

				CTheGameOfLife(int width, int height);
				~CTheGameOfLife();
				!CTheGameOfLife();

			};
		}
	}
}