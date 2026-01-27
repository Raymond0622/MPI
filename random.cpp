#include <atomic>
#include <barrier>
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>

constexpr int NTHREADS = 2;
constexpr int N = 100'000'000;

std::atomic<int> count{0};
std::barrier sync_point(NTHREADS + 1); // workers + main

void worker() {
    // wait until all threads are ready
    sync_point.arrive_and_wait();

    // work
    for (int i = 0; i < N; ++i) {
        count.fetch_add(1, std::memory_order_relaxed);
    }

    // signal completion
    sync_point.arrive_and_wait();
}


void increment() {
    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; ++i)
        threads.emplace_back(worker);

    // release workers
    sync_point.arrive_and_wait(); // here only the main thread joins, but
    // in the worker() barrier, two worker threads have arrived, so for a 
    // total of 3 threads which allows the barrier to be passed now
    // this simulates the fact that all 3 threads have arrived.
    auto start = std::chrono::steady_clock::now();

    // wait for completion
    sync_point.arrive_and_wait();
    auto end = std::chrono::steady_clock::now();

    for (auto& t : threads)
        t.join();

    std::chrono::duration<double> dt = end - start;
    std::cout << "Time: " << dt.count() << " s\n";
}


int main() {
   
    increment();
    std::cout << count;
  
}
