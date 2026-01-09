#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <cmath>
#include <unordered_set>
#include <type_traits>
#include <tuple>

#include "Graph.hpp"
#include "UnionFind.hpp"
#include "RandomTree.hpp"
#include "RandomBinaryTree.hpp"

template <typename T, size_t N>
class Message {
    T msg[N];
};

template <typename T, size_t N>
struct Data {
    int source;
    Message<T, N> msg;
    // require default constructor for creating custom MPI data struct
    Data() {};
    Data(int s) : source(s) {};
};

using DataType = std::variant<Data<int, 100>, Data<int, 200>, 
    Data<double, 20000>, Data<long long, 50>>;

constexpr size_t numType = std::variant_size_v<DataType>;

int getQueueSize(std::vector<std::vector<int>>& recvcount) {
    int tot = 0;
    for (int i = 0; i < numType;i++) {
        for (int j = 0; j < recvcount.size();j++) {
            tot += recvcount[i][j];
        }
    }
    return tot;
};

template <typename T>
void create_mpi_struct(std::size_t idx, T tmp, std::vector<MPI_Datatype>& datatype,
    MPI_Datatype types[4][2], int lengths[4][2]) {
        MPI_Aint base;
        MPI_Aint offset[2];
        MPI_Datatype MPI_Data;
        MPI_Get_address(&tmp, &base);
        MPI_Get_address(&tmp.source, &offset[0]);
        MPI_Get_address(&tmp.msg, &offset[1]);
        offset[0] -= base; offset[1] -= base;
        MPI_Type_create_struct(2, lengths[idx], offset, types[idx], &MPI_Data);
        MPI_Type_commit(&MPI_Data);
        datatype.push_back(MPI_Data);
}

template <std::size_t... Idx>
void create_my_mpi_types(std::vector<MPI_Datatype>& datatype, 
    MPI_Datatype types[4][2], 
    int lengths[4][2], std::integer_sequence<std::size_t, Idx...>) {
    
    // very hacky way of getting each alternative type.
    std::tuple<std::variant_alternative_t<Idx, DataType>...> tup(
        std::variant_alternative_t<Idx, DataType>{}...);
        
    (create_mpi_struct(Idx, std::get<Idx>(tup), datatype, types, lengths), ...);
}


int findIdxType(DataType& type) {
    return type.index();
}

void bfs(Graph& graph, int& rank, int& size, std::vector<MPI_Datatype>& datatype) {
    // now since we have different types, we need sendcount, recvount for each type
    std::vector<std::vector<int>> sendcount, recvcount;

    // senddata[i][j] represents vector jth variant data to send to ith processor
    std::vector<std::vector<std::vector<DataType>>> senddata;
    // // find out how many verticies (based on rank) are going to the next BFS level
    // // This is sort of like degree of a veritices per level

    // // allocate sizes where recbuf[i][j] represent the buffer coming
    // // from the jth rank processor of type ith variant.
    std::vector<std::vector<std::vector<DataType>>> recbuf(numType, std::vector<std::vector<DataType>>(size));
    //using StartType = std::variant_alternative_t<1, DataType>();
    recbuf[2][rank].push_back(Data<double, 20000>(rank));
    int depth = 0;
   
    int global_flag = 0;

    while (true) {

        int local_flag = 0;
        sendcount.assign(numType, std::vector<int>(size, 0));
        recvcount.assign(numType, std::vector<int>(size, 0));
        senddata.clear();
        senddata.assign(size, std::vector<std::vector<DataType>>(numType));
                
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                // data of type ith variant from jth processor
                for (auto& msg : recbuf[i][j]) {
                    // need to get index before visiting
                    int idx_type = findIdxType(msg);
                    std::visit([&](auto data) {
                            int u = data.source;
                            for (auto v : graph.adjList[u]) {
                                int own = graph.getOwner(v);
                                //printf("Rank %d has %d\n", rank, v);
                                sendcount[i][own]++;
                                auto new_data = data;
                                new_data.source = v;
                                senddata[i][own].push_back(std::move(new_data));
                            }
                    }, msg);
                }   
            }
        }

        // // Send the degree to each process
        // since we need to send count data fro each type, we do for each type of variant
        for (int i = 0; i < numType;i++) {
            MPI_Alltoall(sendcount[i].data(), 1, MPI_INT,
                recvcount[i].data(), 1, MPI_INT, MPI_COMM_WORLD);
        }
        recbuf.clear();
        recbuf.assign(numType, std::vector<std::vector<DataType>>(size));
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                recbuf[i][j].resize(recvcount[i][j]);
            }
        }
        // // Post non-blocking receives
        std::vector<MPI_Request> requests;
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                MPI_Request req;
                if (recvcount[i][j] > 0) {
                    MPI_Irecv(recbuf[i][j].data(), recvcount[i][j],
                        datatype[i], j, i, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                }  
            }
        }
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                if (sendcount[i][j] > 0) {
                    MPI_Request req;
                    MPI_Isend(senddata[i][j].data(), sendcount[i][j], 
                        datatype[i], j, i, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                }
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        depth++;
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                for (auto& v : recbuf[i][j]) {
                    std::visit([&](auto& msg) {
                        //printf("Rank %d gets vertex %d of msg type %d with depth %d\n", rank, msg.source, i, depth);
                    }, v);
                }
            }
        };
        //printf("rank %d %d\n", rank, getQueueSize(recbuf));
        if (getQueueSize(recvcount) == 0) {
            local_flag = 1;
        }
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, 
                MPI_LAND, MPI_COMM_WORLD);
        if (global_flag) {
            break;
        }   
        //return;
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = std::stoi(argv[1]);

    // random generator to generate random edges
    Graph graph(n, size);
    if (rank == 0) {
        // this is object slicing, but graph/RandomGraph contains 
        // same member variable, but only rank 0 should fill
        // the member variables up.
        graph = RandomTree(n, size);
    }
   
    MPI_Bcast(&graph.numP, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.ownership.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < n;i++) {
        size_t sizeAdj;
        if (rank == 0) {
            sizeAdj = graph.adjList[i].size();
        }
        MPI_Bcast(&sizeAdj, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        std::vector<int> buffer;
        // iterators do not necessary point to contigous memory, so much transfer it
        // some datat structure that is
        if (rank == 0)
            buffer.assign(graph.adjList[i].begin(), graph.adjList[i].end());
        else
            buffer.resize(sizeAdj);
        MPI_Bcast(buffer.data(), sizeAdj, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            graph.adjList[i].insert(buffer.begin(), buffer.end());
        }
    }
    // prepare MPI stuff for my own data structs
    MPI_Datatype types[4][2] = {{MPI_INT, MPI_INT}, {MPI_INT, MPI_INT},
            {MPI_INT, MPI_DOUBLE}, {MPI_INT, MPI_LONG_LONG}};
    int lengths[4][2] = {{1, 100}, {1, 200}, {1, 20}, {1, 50}};
    std::vector<MPI_Datatype> mydatatype;

    create_my_mpi_types(mydatatype, types, lengths, 
       std::make_integer_sequence<std::size_t, numType>{});
    
    bfs(graph, rank, size, mydatatype);
    for (int i = 0; i < numType;i++) {
        MPI_Type_free(&mydatatype[i]);
    }
    
    MPI_Finalize();
}