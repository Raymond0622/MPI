#ifndef UNIONFIND_H
#define UNIONFIND_H

#include <vector>

struct Union {
    Union(int n);
    int find(int x);
    void unite(int x, int y);
    bool checker(int x, int y);

    int size;
    std::vector<int> parent;
};

#endif