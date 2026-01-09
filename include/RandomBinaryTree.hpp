#ifndef RANDOM_BINARY_TREE_H
#define RANDOM_BINARY_TREE_H
#include <iostream>
#include <vector>
#include "Graph.hpp"

struct RandomBinaryTree : Graph {
    RandomBinaryTree(size_t n, size_t _numP);
    int getParent(int x);
    int getDepth(int x);
    std::vector<int> parent;
    std::vector<int> depth;
};

#endif