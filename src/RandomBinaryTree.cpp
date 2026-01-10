
#include <random>
#include <algorithm>
#include "Graph.hpp"
#include "HelperFunctions.hpp"
#include "RandomBinaryTree.hpp"

    
RandomBinaryTree::RandomBinaryTree(size_t n, size_t _numP) : Graph(n, _numP) {
    parent.resize(n);
    depth.resize(n);
    perm.resize(n);
    // first create random permutation
    // then act like the perm forms the binary tree 
    // in canonical form.
    std::random_device ran;
    std::mt19937 device(ran());
    for (int i= 0; i < n;i++) {
        perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), device);
    std::cout << perm[0] << std::endl;
    for (int i = 0; i < n;i++) {
        depth[perm[i]] = std::log2(i);
        if (2*i + 1 < n) {
            insertEdge(perm[i], perm[2*i + 1]);
            parent[perm[2*i + 1]] = perm[i];
            printf("%d %d\n", perm[i], perm[2*i + 1]);
        }
        if (2*i + 2 < n) {
            insertEdge(perm[i], perm[2*i + 2]);
            parent[perm[2*i + 2]] = perm[i];
            //printf("%d %d\n", perm[i], perm[2*i + 2]);
        }
    }
}
int RandomBinaryTree::getParent(int x) {
    return parent[x];
}
int RandomBinaryTree::getDepth(int x) {
    return depth[x];
}
