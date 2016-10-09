#include "ConcurrentQueue.h"

#ifndef CONCURRENT_QUEUE
#define CONCURRENT_QUEUE

namespace CellUniverse {
	namespace Models {
		namespace CUDA {

			template <typename T> void ConcurrentQueue<T>::Enqueue(T &item) {
				mtx.lock();
				impl.push(item);
				mtx.unlock();
			}

			template <typename T> bool ConcurrentQueue<T>::TryDequeue(T &result) {

				if (mtx.try_lock()) {

					if (!impl.empty()) {
						result = impl.front();
						impl.pop();
						mtx.unlock();
						return true;
					}
					mtx.unlock();
					return false;
				}
				return false;
			}

			template <typename T> unsigned ConcurrentQueue<T>::Count() {
				mtx.lock();
				unsigned result = impl.size();
				mtx.unlock();
				return result;
			}

			template <typename T> bool ConcurrentQueue<T>::IsEmpty() {
				mtx.lock();
				bool result = impl.empty();
				mtx.unlock();
				return result;
			}

			template <typename T> ConcurrentQueue<T>::ConcurrentQueue() { }
			template <typename T> ConcurrentQueue<T>::~ConcurrentQueue() { }
		}
	}
}

#endif // !CONCURRENT_QUEUE