#include <random>
#include <chrono>

#include "HelperFunctions.hpp"

std::mt19937 RandomSeed() {
    const auto curr = std::chrono::system_clock::now();
    const auto epoch = curr.time_since_epoch();
    unsigned int seed = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
    std::mt19937 engine(seed);
    return engine;
}

