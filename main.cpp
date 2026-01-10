#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <cmath>
#include <unordered_set>
#include <type_traits>
#include <tuple>
#include <string>

#include "Graph.hpp"
#include "UnionFind.hpp"
#include "RandomTree.hpp"
#include "RandomDAG.hpp"
#include "RandomBinaryTree.hpp"

template <typename T, size_t N>
struct Message {
    T msg[N];
};

template <typename T, size_t N>
struct Data {
    int source;
    T msg[N];
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
        MPI_Get_address(&tmp.msg[0], &offset[1]);
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


// this creates a tuple that stores vectors of each variant type
// since we cant store all the variants in a single tensor, 
// we have to store them separately, but nicely into one single struct
template <typename U>
struct Tuple;

template <size_t... Idx>
struct Tuple <std::integer_sequence<size_t, Idx...>> {
    Tuple(int size) {
        resize(size);
    }
    std::tuple<std::vector<std::vector<std::variant_alternative_t<Idx, DataType>>>...> tup;
    void resize(int size) {
        (std::get<Idx>(tup).resize(size), ...);
    }
    void insert(int rank, 
        int source, size_t idx, int count) {
        // can use lambdas (C++20)
        auto f = [&]<size_t I>() {
            if (I == idx) {
                for (int j = 0; j < count;j++) {
                    using T = typename std::remove_reference_t<decltype(std::get<I>(tup)[rank])>::value_type;
                    std::get<I>(tup)[rank].push_back(
                        T(source));
                }
            }
        };
        (f.template operator()<Idx>(), ...);
    }
};

// how many to insert for which message type and how many
// idx is the type of message
// rank is the which processor to put it into (2nd idx in tuple)
// source is the vertex the message will be sent to
// this is to initialize the initial data messages
// this seems a bit disgusting, but good metaprogramming practice
template <size_t... Idx>
void insert(Tuple<std::integer_sequence<size_t, Idx...>>& Tup, int rank, 
    int source, size_t idx, int count) {
    // can use lambdas (C++20)
    auto f = [&]<size_t I>() {
        if (I == idx) {
            for (int j = 0; j < count;j++) {
                using T = typename std::remove_reference_t<decltype(std::get<I>(Tup.tup)[rank])>::value_type;
                std::get<I>(Tup.tup)[rank].push_back(
                    T(source));
            }
        }
    };
    (f.template operator()<Idx>(), ...);
}

// overloaded version of insert, but now we insert
// the input contains data
template <typename T, size_t... Idx>
void insert(Tuple<std::integer_sequence<size_t, Idx...>>& Tup, int rank, 
    T&& data, size_t idx) {
    // can use lambdas (C++20)
    auto f = [&]<size_t I>() {
        if (I == idx) {
            std::get<I>(Tup.tup)[rank].push_back(
                std::forward<T>(data));
        }
    };
    (f.template operator()<Idx>(), ...);
}

template <size_t... Idx>
void clear(Tuple<std::integer_sequence<size_t, Idx...>>& Tup) {
    auto f = [&]<size_t I>() {
        std::get<I>(Tup.tup).clear();
    };
    (f.template operator()<Idx>(), ...);
}

// resize the inner recbuf, this is for resizing
// actual data array
// count contains the # of idx type data message will get
// for each process
// so count[i] represents # idx msg type the it process has.
template <size_t... Idx>
void resize_inner(Tuple<std::integer_sequence<size_t, Idx...>>& Tup, 
                        const int& idx, const std::vector<int>& count) {
    auto f = [&]<size_t I>() {
        if (I == idx) {
            auto& tmp = std::get<I>(Tup.tup);
            for (int j = 0; j < count.size();j++) {
                tmp[j].resize(count[j]);
            }
        }
    };
    (f.template operator()<Idx>(), ...);
}

// resize the outer array so that 
// each element of the tuple (which is a 2D array)
// has size equal to # of processor
template <size_t... Idx>
void resize_outer(Tuple<std::integer_sequence<size_t, Idx...>>& Tup, 
                    const int& size) {
    auto f = [&]<size_t I>() {
        std::get<I>(Tup.tup).resize(size);
    };
    (f.template operator()<Idx>(), ...);
}


template <size_t... Idx>
void bfs(Graph& graph, int& rank, int& size, std::vector<MPI_Datatype>& datatype,
    Tuple<std::integer_sequence<size_t, Idx...>>& recbuf) {
    // now since we have different types, we need sendcount, recvount for each type
    std::vector<std::vector<int>> sendcount, recvcount;

    // std::get<i>(senddata)[j] represents vector ith variant data to send to jth processor
    Tuple<std::integer_sequence<size_t, Idx...>> senddata;
    // // find out how many verticies (based on rank) are going to the next BFS level
    // // This is sort of like degree of a veritices per level

    // // allocate sizes where std::get<i>(recbuf)[j] represent the buffer coming
    // // from the jth rank processor of type ith variant.
    int depth = 0;
   
    int global_flag = 0;

    while (true) {

        int local_flag = 0;
        sendcount.assign(numType, std::vector<int>(size, 0));
        recvcount.assign(numType, std::vector<int>(size, 0));
        //senddata.clear();
       // senddata.assign(size, std::vector<std::vector<DataType>>(numType));
        clear(senddata);
        resize_outer(senddata, size);
                
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                // data of type ith variant from jth processor
                for (auto& msg : recbuf[i][j]) {
                    // need to get index before visiting
                    // we already know the index due to structure
                    // of recbuf, but lets do it this way
                    int idx_type = findIdxType(msg);
                    std::visit([&](auto data) {
                            int u = data.source;
                            int own;
                            for (auto v : graph.adjList[u]) {
                                own = graph.getOwner(v);
                                //printf("%d to %d\n", u, v);
                                sendcount[i][own]++;
                                auto new_data = data;
                                new_data.source = v;
                                insert(recbuf, own, std::move(new_data), i);
                                // std::visit([&](auto& d) {
                                //     if (depth == 1)
                                //         printf("%d %d\n", rank, d.source);
                                // }, senddata[i][own].back());
                            }
                            
                    }, msg);
                   
                }   
            }
        }
        // for (int i = 0;i < numType;i++) {
        //     for (int j = 0; j < size;j++) {
        //         if (depth == 1 && sendcount[i][j] > 0)
        //             printf("sendcount %d %d %d\n", rank, j, sendcount[i][j]);
        //     }
        // }

        // // Send the degree to each process
        // since we need to send count data fro each type, we do for each type of variant
        for (int i = 0; i < numType;i++) {
            MPI_Alltoall(sendcount[i].data(), 1, MPI_INT,
                recvcount[i].data(), 1, MPI_INT, MPI_COMM_WORLD);
        }
        //recbuf.clear();
        //recbuf.assign(numType, std::vector<std::vector<DataType>>(size));
        clear(recbuf);
        resize_outer(recbuf, size);
        for (int i = 0; i < numType;i++) {
            resize_inner(recbuf, i, recvcount[i]);
        }

        // for (int i = 0;i < numType;i++) {
        //     for (int j = 0; j < size;j++) {
        //         if (depth == 1 && recvcount[i][j] > 0)
        //             printf("recvcount %d %d %d\n", rank, j, recvcount[i][j]);
        //     }
        // }
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
        if (rank == 0) {
            printf("FINSIHED LEVEL\n");
        }
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                for (auto& v : recbuf[i][j]) {
                    std::visit([&](auto& msg) {
                        //printf("Rank %d gets vertex %d of msg type %d with depth %d\n", rank, msg.source, i, depth);
                        if (depth == 2)
                            printf("check: %d\n", msg.source);
                    }, v);
                }
            }
        };

        if (depth == 2) return;

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

    const std::string GRAPH = std::string(GRAPH_TYPE);
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
        if (GRAPH == "BT") graph = RandomBinaryTree(n, size);
        else if (GRAPH == "RT") graph = RandomTree(n, size);
        else if (GRAPH == "DAG") graph = RandomDAG(n, size);
    }
   
    MPI_Bcast(&graph.numP, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.ownership.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.perm.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
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

    using Seq = decltype(std::make_integer_sequence<std::size_t, numType>{});
    create_my_mpi_types(mydatatype, types, lengths, 
       Seq{});
    
    Tuple<Seq> recbuf(size);

    if (GRAPH == "BT") {
        // get the root node's process owner
        //printf("AYA");
        int owner = graph.getOwner(graph.perm[0]);
        //printf("Owner node: %d %d\n", graph.perm[0], owner);
        insert(recbuf, owner, graph.perm[0], 0, 1);
    }
    else if (GRAPH == "RT" || GRAPH == "DAG") {
        //printf("AYA");
        insert(recbuf, rank, rank, 0, 2);
        //recbuf[2][rank].push_back(Data<double, 20000>(rank));
    }
    //bfs(graph, rank, size, mydatatype, recbuf);
    for (int i = 0; i < numType;i++) {
        MPI_Type_free(&mydatatype[i]);
    }
    
    MPI_Finalize();
}