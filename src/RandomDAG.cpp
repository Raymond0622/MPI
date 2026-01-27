
#include <random>
#include <algorithm>
#include <chrono>
#include "Graph.hpp"
#include "RandomDAG.hpp"
#include "HelperFunctions.hpp"


RandomDAG::RandomDAG(size_t n, size_t _numP) : Graph(n, _numP) {
    std::random_device ran;
    std::mt19937 device(ran());
    for (int i= 0; i < n;i++) {
        perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), device);
    auto engine = RandomSeed();
    std::vector<std::vector<int>> edges;
    // edges.push_back({3, 2});
    // edges.push_back({4, 2});
    // edges.push_back({4, 3});
    // edges.push_back({6, 2});
    // edges.push_back({5, 3});
    // edges.push_back({5, 4});
    // edges.push_back({1, 5});
    // edges.push_back({1, 6});
    // edges.push_back({0, 5});
    // edges.push_back({0, 3});
    // for (auto& p : edges) {
    //     insertEdge(p[0], p[1]);
    //     //printf("%d %d\n", p[0], p[1]);
    // }
    for (int i = 0; i < n;i++) {
        if (i == 0) {
            continue;
        }
        for (int j = 0; j < 2;j++) {
            std::uniform_int_distribution<int> dist(0, i - 1);
            int p = dist(engine);
            insertEdge(perm[i], perm[p]);
            //printf("%d %d\n", perm[i], perm[p]);
        }
    }
}