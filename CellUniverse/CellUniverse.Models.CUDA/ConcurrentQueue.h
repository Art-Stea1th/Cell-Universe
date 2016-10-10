#pragma once
#include <mutex>
#include <queue>

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			template<typename T> class ConcurrentQueue {

			private:

				std::recursive_mutex rmtx;
				std::queue<T> impl;

			public:

				void Enqueue(T &item);
				bool TryDequeue(T &result);

				unsigned Count();
				bool IsEmpty();

				ConcurrentQueue();
				~ConcurrentQueue();
			};
		}
	}
}
#include "ConcurrentQueue.cu"