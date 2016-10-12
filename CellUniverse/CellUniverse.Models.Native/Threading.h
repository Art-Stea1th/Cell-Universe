#pragma once
#include <thread>
#include <chrono>

#define SLEEP(n) std::this_thread::sleep_for(std::chrono::milliseconds(n))