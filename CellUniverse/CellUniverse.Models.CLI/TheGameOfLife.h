#pragma once
#define CLI_MODEL

namespace CellUniverse {
	namespace Models {

		namespace Native { namespace CPP { class CUniverse; } }

		namespace CLI {

			public ref class CTheGameOfLife {

			private:

				Native::CPP::CUniverse * _impl;

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