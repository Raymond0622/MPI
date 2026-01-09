#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <vector>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int local = rank + 1;
    int prefix = 0;
    MPI_Scan(&local, &prefix, 1, MPI_INT,
        MPI_SUM, MPI_COMM_WORLD);

    std::cout << prefix << std::endl;
    MPI_Finalize();

}