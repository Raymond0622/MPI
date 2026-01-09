#include <mpi.h>
#include <iostream>
#include <vector>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int right = rank + 1 >= size ? 0 : rank + 1;
    int left = rank - 1 < 0 ? size - 1 : rank - 1;
    int rec;
    int send = rank;
    int global = 0;
    MPI_Sendrecv(&send, 1, MPI_INT, right, 0, 
        &rec, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Allreduce(&rec, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Rank : " << rank << " " << global << std::endl;

    MPI_Finalize();
}