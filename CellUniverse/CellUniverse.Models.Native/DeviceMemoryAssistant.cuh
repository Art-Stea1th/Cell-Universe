#include "DefinesCUDA.cuh"


namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CUDA {

				HT void AllocateVectorOnDevice(bool* &dVector, int alignedBlocksCount, int threadsPerBlock);
				GL void InitializeVectorOnDevice(bool* dVector, int alignedLength);
				HT void FreeVectorOnDevice(bool* dVector);

			}
		}
	}
}