#include "Graph.hpp"
#include <iostream>
#include <unordered_set>

Graph::Graph (size_t n, size_t _numP) {
    adjList.resize(n);
    ownership.resize(n);
    numP = _numP;
    for (int i = 0; i < n;i++) {
        ownership[i] = i % numP;
    }
}
// denotes directed edge from p to q
void Graph::insertEdge(int p, int q) {
    adjList[p].insert(q);
}
    // denote removal of an edge from p to q;
void Graph::removeEdge(int p, int q) {
    if (adjList[p].count(q)) {
        adjList[p].erase(q);  
    }
}
int Graph::getOwner(size_t n) {
    return ownership[n];
}