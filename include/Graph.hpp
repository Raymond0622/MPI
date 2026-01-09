#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <unordered_set>

struct Graph {
    Graph(size_t n, size_t _numP);
    void insertEdge(int p, int q);
    void removeEdge(int p, int q);
    int getOwner(size_t n);

    std::vector<std::unordered_set<int>> adjList;
    std::vector<int> ownership;
    size_t numP;
};

#endif 

