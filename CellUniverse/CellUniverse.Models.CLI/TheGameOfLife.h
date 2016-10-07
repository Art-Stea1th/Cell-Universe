#pragma once
#define CLI_MODEL

namespace CellUniverse {
	namespace Models {

		namespace CUDA { class CTheGameOfLife; }

		namespace CLI {

			public ref class CTheGameOfLife {

				CUDA::CTheGameOfLife * _impl;

			public:

				bool * GetNextGeneration();

				void Destroy();

				CTheGameOfLife(int width, int height);
				~CTheGameOfLife();
				!CTheGameOfLife();

			};
		}
	}
}