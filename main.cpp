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
    Data<double, 200>, Data<long long, 50>>;

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
    MPI_Datatype types[numType][2], 
    int lengths[numType][2], std::integer_sequence<std::size_t, Idx...>) {
    
    // very hacky way of getting each alternative type.
    std::tuple<std::variant_alternative_t<Idx, DataType>...> tup(
        std::variant_alternative_t<Idx, DataType>{}...);
        
    (create_mpi_struct(Idx, std::get<Idx>(tup), datatype, types, lengths), ...);
}


int findIdxType(DataType& type) {
    return type.index();
}

// need this struct to extract template types for Data
// i.e., we say that T = Data<int, 200>, 
// how do we get T = int, and N = 200?
template <typename T>
struct extract;

template <typename T, size_t N>
struct extract<Data<T, N>> {
    using type = T;
    constexpr static int value = N;
};


template <size_t... Idx>
void extract_f(std::integer_sequence<size_t, Idx...>, int (&lengths)[numType][2]) {
    auto f = [&]<size_t I>() {
        lengths[I][0] = 1;
        lengths[I][1] = extract< typename std::variant_alternative_t<I, DataType>>::value;
    };
    (f.template operator()<Idx>(), ...);
    
}

// this creates a tuple that stores vectors of each variant type
// since we cant store all the variants in a single tensor, 
// we have to store them separately, but nicely into one single struct
template <typename U>
struct Tuple;

template <size_t... Idx>
struct Tuple <std::integer_sequence<size_t, Idx...>> {
    std::tuple<std::vector<std::vector<std::variant_alternative_t<Idx, DataType>>>...> tup;
    Tuple(int size) {
        resize(size);
    }
    // resize the outer array so that 
    // each element of the tuple (which is a 2D array)
    // has size equal to # of processor
    void resize(int size) {
        (std::get<Idx>(tup).resize(size), ...);
    }
    // how many to insert for which message type and how many
    // idx is the type of message
    // rank is the which processor to put it into (2nd idx in tuple)
    // source is the vertex the message will be sent to
    // this is to initialize the initial data messages
    // this seems a bit disgusting, but good metaprogramming practice
    void insert(int rank, int source, size_t idx, int count) {
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
    void clear() {
        auto f = [&]<size_t I>() {
            std::get<I>(tup).clear();
        };
        (f.template operator()<Idx>(), ...);
    }
    // resize the inner recbuf, this is for resizing
    // actual data array
    // count contains the # of idx type data message will get
    // for each process
    // so count[i] represents # idx msg type the it process has.
    void resize_inner(const int& idx, const std::vector<int>& count) {
        auto f = [&]<size_t I>() {
            if (I == idx) {
                auto& tmp = std::get<I>(tup);
                for (int j = 0; j < count.size();j++) {
                    tmp[j].resize(count[j]);
                }
            }
        };
        (f.template operator()<Idx>(), ...);
    }
    // get size of inner, i.e., size of the idx message of jth processor
    int size(const int& idx, const int& j) {
        int result;
        auto f = [&]<size_t I>() {
            if (I == idx) {
                result = std::get<I>(tup)[j].size();
            }
        };
        (f.template operator()<Idx>(), ...);
        return result;
    }
    // print the message of the rank-th processor, with size # of MPI processors
    // with current depth
    void print(const int& rank, const int& size, const int& depth) {
        auto f = [&]<size_t I>() {
            for (int j =0; j < size;j++) {
                auto tmp = static_cast<std::variant_alternative_t<I, DataType>*>(data(I, j));
                for (int k = 0; k < Tuple::size(I, j);k++) {
                    printf("Rank %d gets vertex %d of msg type %lu with depth %d\n", 
                            rank, (*(tmp + k)).source, I, depth);
                }
            }
        };
        (f.template operator()<Idx>(), ...);
    }
        // grabs pointer to the ith msg type of jth processor
    auto data(const int& i, const int& j) {
        void* result = nullptr;
        auto f = [&]<size_t I>() {
            if (I == i) {
                result = std::get<I>(tup)[j].data();
            }
        };
        (f.template operator()<Idx>(), ...);
        return result;
    }
    // overloaded version of insert, but now we insert
    // the input contains data
    // this needs to be constexpr compiler still checks over
    // each fold (fold expression) regardless if the branch is
    // is fufilled, to avoid this used constexpr
    template <typename T, size_t N>
    void insert(int rank, T&& data, std::integral_constant<size_t, N>) {
        // can use lambdas (C++20)
        auto f = [&]<size_t I>() {
            if constexpr (I == N) {
                std::get<I>(tup)[rank].push_back(
                    std::forward<T>(data));
            }
        };
        (f.template operator()<Idx>(), ...);
    }
    // push data from recbuf to senddata, this enacts the BFS action
    // transports message from current level to next level
    // Current object owns the current level, and data 
    // is storage for the next level
    void push(Tuple<std::integer_sequence<size_t, Idx...>>& data, 
            std::vector<std::vector<int>>& sendcount, Graph& graph, const int& size) {
        auto f = [&]<size_t I>() {
            for (size_t j = 0; j < size;j++) {
                auto tmp = static_cast<std::variant_alternative_t<I, DataType>*>(Tuple::data(I, j));
                for (int k = 0; k < Tuple::size(I, j);k++) {
                    auto new_data = (*(tmp + k));
                    int& u = new_data.source;
                    int own;
                    for (auto v : graph.adjList[u]) {
                        new_data.source = v;
                        own = graph.getOwner(v);
                        sendcount[I][own]++;
                        data.insert(own, std::move(new_data), std::integral_constant<size_t, I>{});
                    }
                }
            }
        };
        (f.template operator()<Idx>(), ...);
    }
};

template <size_t... Idx>
void bfs(Graph& graph, int& rank, int& size, std::vector<MPI_Datatype>& datatype,
    Tuple<std::integer_sequence<size_t, Idx...>>& recbuf) {
    // now since we have different types, we need sendcount, recvount for each type
    std::vector<std::vector<int>> sendcount, recvcount;

    // std::get<i>(senddata)[j] represents vector ith variant data to send to jth processor
    Tuple<std::integer_sequence<size_t, Idx...>> senddata(size);
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
        senddata.clear();
        senddata.resize(size);

        recbuf.push(senddata, sendcount, graph, size);
                
        // for (int i = 0;i < numType;i++) {
        //     for (int j = 0; j < size;j++) {
        //         if (sendcount[i][j] > 0)
        //             printf("sendcount %d %d %d %d\n", i, rank, j, sendcount[i][j]);
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
        recbuf.clear();
        recbuf.resize(size);
        for (int i = 0; i < numType;i++) {
            recbuf.resize_inner(i, recvcount[i]);
        }

        // for (int i = 0;i < numType;i++) {
        //     for (int j = 0; j < size;j++) {
        //         if (recvcount[i][j] > 0)
        //             printf("recvcount %d %d %d\n", rank, j, recvcount[i][j]);
        //     }
        // }
        // // Post non-blocking receives
        std::vector<MPI_Request> requests;
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                MPI_Request req;
                if (recvcount[i][j] > 0) {
                    MPI_Irecv(recbuf.data(i, j), recvcount[i][j],
                        datatype[i], j, i, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                }  
            }
        }
        for (int i = 0; i < numType;i++) {
            for (int j = 0; j < size;j++) {
                if (sendcount[i][j] > 0) {
                    MPI_Request req;
                    MPI_Isend(senddata.data(i, j), sendcount[i][j], 
                        datatype[i], j, i, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                }
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        depth++;
        recbuf.print(rank, size, depth);
                
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
   
    // cast this to other processors in order for the random graph 
    // to be the same across 
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
    // extract template types from DataType defintion
    MPI_Datatype types[numType][2] = {{MPI_INT, MPI_INT}, {MPI_INT, MPI_INT},
            {MPI_INT, MPI_DOUBLE}, {MPI_INT, MPI_LONG_LONG}};

    int lengths[numType][2];
    extract_f(std::make_integer_sequence<size_t, numType>{}, lengths);
    std::vector<MPI_Datatype> mydatatype;

    using Seq = decltype(std::make_integer_sequence<std::size_t, numType>{});
    create_my_mpi_types(mydatatype, types, lengths, 
       Seq{});
    
    Tuple<Seq> recbuf(size);

    if (GRAPH == "BT") {
        // get the root node's process owner
        int owner = graph.getOwner(graph.perm[0]);
        //printf("Owner node: %d %d\n", graph.perm[0], owner);
        recbuf.insert(owner, graph.perm[0], 0, 1);
    }
    else if (GRAPH == "RT" || GRAPH == "DAG") {
        recbuf.insert(rank, rank, 2, 3);
        recbuf.insert(rank, rank, 1, 1);
        //recbuf[2][rank].push_back(Data<double, 20000>(rank));
    }
    bfs(graph, rank, size, mydatatype, recbuf);
    for (int i = 0; i < numType;i++) {
        MPI_Type_free(&mydatatype[i]);
    }
    
    MPI_Finalize();
}