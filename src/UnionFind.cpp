#include "UnionFind.hpp"

Union::Union(int n) : size(n) {
    parent.resize(n, 0);
    for (int i = 0; i < n;i++) {
        parent[i] = i;
    }
};
int Union::find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}
void Union::unite(int x, int y) {
    int p1 = find(x), p2 = find(y);
    if (p1 != p2) {
        parent[p1] = p2;
    }
}
bool Union::checker(int x, int y) {
    return find(x) == find(y);
}