#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm commRow, commCol;
    int p = (int) std::sqrt(size);
    MPI_Comm_split(MPI_COMM_WORLD, rank / p, rank % p, &commRow);
    MPI_Comm_split(MPI_COMM_WORLD, rank % p, rank / p, &commCol);
    int local = rank + 1;
    int row_sum = 0;
    std::vector<int> row_sums;
    row_sums.resize(p);
    MPI_Allreduce(&local, &row_sum, 1, MPI_INT, 
        MPI_SUM, commRow);
    if (rank % p == 0) {
        MPI_Send(&row_sum, 1, MPI_INT, 
            0, 0, MPI_COMM_WORLD);
        printf("Rank %d : row sum %d\n", rank, row_sum);
    }
    int final_sum= 0;
    if (rank == 0) {
        int meh, meh2;
        MPI_Recv(&meh, 1, MPI_INT, 0,
        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(&meh2, 1, MPI_INT, 2,
        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        final_sum += meh + meh2;
        std::cout << final_sum << std::endl;
    }
    MPI_Finalize();
}