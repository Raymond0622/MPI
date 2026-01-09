#include "Graph.hpp"
#include "UnionFind.hpp"
#include "RandomTree.hpp"
#include "HelperFunctions.hpp"
#include <vector>
#include <chrono>
#include <random>

RandomTree::RandomTree(size_t n, size_t _numP) : Graph(n, _numP) {
    Union union_find(n);
    
    auto engine = RandomSeed();

    // create graph so that we dont have cycles. Use union find algorthim to find cycle
    for (int i = 0;i < n- 1;i++) {
        std::uniform_int_distribution<int> diststart(0, numP-1);
        std::uniform_int_distribution<int> dist(0, n - 1);
        while (true)  {
            int p = dist(engine);
            int q = dist(engine);
            
            if (adjList[p].count(q) || p == q || union_find.checker(p, q)) {
                continue;
            }
            else {
                insertEdge(p, q);
                printf("%d %d %d %d\n", i, p, q, union_find.checker(p, q));
                union_find.unite(p, q);
                break;
            }
        }
    }
}