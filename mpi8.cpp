#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <random>
#include <unordered_set>

template <typename T, size_t N>
class Message {
    T msg[N];
    size_t len = N;
    public:
        size_t size() {
            return len;
        }
};

int getQueueSize(std::vector<std::vector<int>>& recbuf) {
    int tot = 0;
    for (int i = 0; i < recbuf.size();i++) {
        tot += recbuf[i].size();
    }
    return tot;
}

struct Union {
    int size;
    std::vector<int> parent;
    Union(int n) : size(n) {
        parent.resize(n, 0);
        for (int i = 0; i < n;i++) {
            parent[i] = i;
        }
    };
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    void unite(int x, int y) {
        int p1 = find(x), p2 = find(y);
        if (p1 != p2) {
            parent[p1] = p2;
        }
    }
    bool checker(int x, int y) {
        return find(x) == find(y);
    }
};

struct Graph {
    Graph (size_t n, size_t _numP) {
        adjList.resize(n);
        ownership.resize(n);
        numP = _numP;
        for (int i = 0; i < n;i++) {
            ownership[i] = i % _numP;
        }
    }
    // denotes directed edge from p to q
    void insertEdge(int p, int q) {
        adjList[p].insert(q);
    }
    // denote removal of an edge from p to q;
    void removeEdge(int p, int q) {
        if (adjList[p].count(q)) {
            adjList[p].erase(q);  
        }
    }
    int getOwner(size_t n) {
        return ownership[n];
    }
    std::vector<std::unordered_set<int>> adjList;
    std::vector<int> ownership;
    size_t numP;
};

int main(int argc, char** argv) {
    int n = 100;

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // random generator to generate random edges
    const auto curr = std::chrono::system_clock::now();
    const auto epoch = curr.time_since_epoch();
    unsigned int seed = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
    std::mt19937 engine(seed);

    // create graph so that we dont have cycles. Use union find algorthim to find cycle
    Union union_find(n);
    Graph graph(n, size);
    for (int i = 0;i < 99;i++) {
        std::uniform_int_distribution<int> diststart(0, 3);
        std::uniform_int_distribution<int> dist(0, 99);
        while (true)  {
            int p = diststart(engine);
            int q = dist(engine);
            //printf("%d %d %d %d\n", i, p, q, union_find.checker(p, q));
            if (graph.adjList[p].count(q) || p == q || union_find.checker(p, q)) {
                continue;
            }
            else {
                graph.insertEdge(p, q);
                union_find.unite(p, q);
                break;
            }
        }
    }
    // Message<char, 100> msg;
    std::vector<int> sendcount, recvcount;
    std::vector<std::vector<int>> sendvertices;
    // // find out how many verticies (based on rank) are going to the next BFS level
    // // This is sort of like degree of a veritices per level

    // // allocate sizes where recbuf[i] represent the buffer coming
    // // from the ith rank processor.
    std::vector<std::vector<int>> recbuf(size);
    // recbuf is the starting point
    recbuf[rank].push_back(rank);
    int depth = 0;
   
    int global_flag = 0;
    while (true) {
        int local_flag = 0;
        sendcount.assign(size, 0);
        recvcount.assign(size, 0);
        sendvertices.clear();
        sendvertices.resize(size);

        for (int i = 0; i < size;i++) {
            for (auto& u : recbuf[i]) {
                for (auto v : graph.adjList[u]) {
                    int own = graph.getOwner(v);
                    //printf("Rank %d has %d\n", rank, v);
                    sendcount[own]++;
                    sendvertices[own].push_back(v);
                }
            }
        }
        // // Send the degree to each process
        MPI_Alltoall(sendcount.data(), 1, MPI_INT, 
            recvcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        recbuf.clear();
        recbuf.resize(size);
        for (int i = 0; i < size;i++) {
            recbuf[i].resize(recvcount[i]);
        }
        // // Post non-blocking receives
        std::vector<MPI_Request> requests;
        for (int i = 0; i < size;i++) {
            MPI_Request req;
            if (recvcount[i] > 0) {
                MPI_Irecv(recbuf[i].data(), recvcount[i],
                    MPI_INT, i, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
            }
            
        }
        for (int i = 0; i < size;i++) {
            if (sendcount[i] > 0) {
                MPI_Request req;
                MPI_Isend(sendvertices[i].data(), sendcount[i], 
                    MPI_INT, i, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        depth++;
        printf("Rank %d has %d vertices %d depth\n", rank, getQueueSize(recbuf), depth);
        // for (auto v : recbuf[i])
        //     printf("Rank %d gets vertex %d\n", rank, v);

        if (getQueueSize(recbuf) ==0) {
            local_flag = 1;
        }
        
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, 
                MPI_LAND, MPI_COMM_WORLD);
    
        if (global_flag) {
            break;
        }
        
    }
    
    MPI_Finalize();

}