#include <mpi.h>
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global;
    int local = rank + 1;
    std::cout << "Rank " << rank << " contributes " << rank + 1 << std::endl;
    MPI_Reduce(&local, &global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Global Sum = " << global;
    }
    MPI_Finalize();
}