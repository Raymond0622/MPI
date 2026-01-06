#include <iostream>
#include <mpi.h>
#include <stdlib.h>

struct Pair {
    int bin;
    int count;
};


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n_local = 100;
    std::vector<int> A;
    for (int i = 0; i < n_local;i++) {
        A.push_back(i);
    }
    std::vector<int> sendcount(size, 0);
    std::vector<std::vector<Pair>> senddata(size);
    for (auto& i : A) {
        sendcount[i % size]++;
        senddata[i % size].push_back({i, A[i]});
    }
    std::vector<int> recvcount(size, 0);
    MPI_Alltoall(sendcount.data(), 1, MPI_INT, recvcount.data(), 1, MPI_INT, 
        MPI_COMM_WORLD);

    std::vector<std::vector<Pair>> recvdata(size);
    for (int i = 0; i < size;i++) {
        recvdata[i].resize(recvcount[i]);
    }
    std::vector<MPI_Request> requests;
    MPI_Datatype MPI_PAIR;
    MPI_Type_contiguous(2, MPI_INT, &MPI_PAIR);
    MPI_Type_commit(&MPI_PAIR);

    for (int i = 0; i < size;i++) {
        if (recvcount[i] > 0) {
            MPI_Request req;
            MPI_Irecv(recvdata[i].data(), recvcount[i], 
                MPI_PAIR, i, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }
    for (int i = 0; i < size;i++) {
        if (sendcount[i] > 0) {
            MPI_Request req;
            MPI_Isend(senddata[i].data(), sendcount[i], MPI_PAIR, 
                    i, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);

    MPI_Type_free(&MPI_PAIR);
    MPI_Finalize();
}