#include "SimpleFSM.h"
#include "Threading.h"

namespace CellUniverse {
	namespace Models {
		namespace Native {
			namespace CPP {

				void SimpleFSM::Subscribe(IFSMCompatible* &invocableSubscriber) {
					invocableSubscribers.push_back(invocableSubscriber);
				}

				SimpleFSM &SimpleFSM::Instance() {
					static SimpleFSM sfsm;
					return sfsm;
				}

				void SimpleFSM::Start() {

					if (state != Terminated) {
						Stop();
					}
					state = Started;
					std::thread thr(&SimpleFSM::StartProcess, this);
					thr.detach();
				}

				void SimpleFSM::StartProcess() {
					while (state == Started) {
						for each (auto subscriber in invocableSubscribers) {							
							subscriber->OnStarted();
						}
					}
					state = Terminated;
				}

				void SimpleFSM::Stop() {
					state = WillBeTerminated;
					while (state != Terminated) { SLEEP(1); }
				}
				

				SimpleFSM &SimpleFSM::operator= (SimpleFSM const &obj) { }

				SimpleFSM::SimpleFSM() { }
				SimpleFSM::SimpleFSM(SimpleFSM const &obj) { }				
				SimpleFSM::~SimpleFSM() { }
			}
		}
	}
}