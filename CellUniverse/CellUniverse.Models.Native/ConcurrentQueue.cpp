#include "ConcurrentQueue.h"

#ifndef CONCURRENT_QUEUE
#define CONCURRENT_QUEUE

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CPP {

				template <typename T> void ConcurrentQueue<T>::Enqueue(T &item) {
					rmtx.lock();
					impl.push(item);
					rmtx.unlock();
				}

				template <typename T> bool ConcurrentQueue<T>::TryDequeue(T &result) {

					if (rmtx.try_lock()) {

						if (!impl.empty()) {
							result = impl.front();
							impl.pop();
							rmtx.unlock();
							return true;
						}
						rmtx.unlock();
						return false;
					}
					return false;
				}

				template <typename T> unsigned ConcurrentQueue<T>::Count() {
					rmtx.lock();
					unsigned result = impl.size();
					rmtx.unlock();
					return result;
				}

				template <typename T> bool ConcurrentQueue<T>::IsEmpty() {
					rmtx.lock();
					bool result = impl.empty();
					rmtx.unlock();
					return result;
				}

				template <typename T> ConcurrentQueue<T>::ConcurrentQueue() { }
				template <typename T> ConcurrentQueue<T>::~ConcurrentQueue() { }
			}
		}
	}
}

#endif // !CONCURRENT_QUEUE