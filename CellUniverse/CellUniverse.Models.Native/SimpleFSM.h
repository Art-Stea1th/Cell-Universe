#pragma once
#include <list>
#include <thread>
#include "IFSMCompatible.h"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CPP {

				class SimpleFSM {

				private:

					enum  FSMState { Started, WillBeTerminated, Terminated };				

					FSMState state;
					std::list<IFSMCompatible*> invocableSubscribers;

					void StartProcess();

					SimpleFSM();
					SimpleFSM(SimpleFSM const& obj);
					SimpleFSM& operator= (SimpleFSM const& obj);

					~SimpleFSM();

				public:

					void Subscribe(IFSMCompatible* &invocableSubscriber);

					static SimpleFSM &Instance();

					void Start();
					void Stop();
				};
			}
		}
	}
}