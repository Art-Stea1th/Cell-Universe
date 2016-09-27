#pragma once


namespace CellUniverse {
	namespace Models {
		namespace Unmanaged {

			class CTheGameOfLifeNative;

			namespace Wrappers {

				public ref class CTheGameOfLifeWrapper {

					Unmanaged::CTheGameOfLifeNative * _impl;

				public:

					bool ** GetNextGeneration();

					void Destroy();

					CTheGameOfLifeWrapper(int width, int height);
					~CTheGameOfLifeWrapper();
					!CTheGameOfLifeWrapper();

				};
			}
		}		
	}	
}