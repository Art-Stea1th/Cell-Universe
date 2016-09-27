#pragma once


namespace CellUniverse {
	namespace Models {
		namespace Unmanaged {

			class __declspec(dllexport) CTheGameOfLifeNative {

			private:

				int _width;
				int _height;

				bool ** _generation;

				void Initialize(const int &width, const int &height);
				void Destroy();


			public:

				bool** GetNextGeneration();

				CTheGameOfLifeNative(const int &width, const int &height);
				~CTheGameOfLifeNative();
			};
		}
	}	
}