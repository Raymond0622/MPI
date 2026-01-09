#include <iostream>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 10;
    std::vector<int> arr(n);
    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank / 2, &newcomm);

    double local = std::accumulate(arr.begin(), arr.end(), 0);
    double maxxy = *std::max_element(arr.begin(), arr.end());
    double globalSum = 0, globalMaxxy = 0;
    MPI_Reduce(&local, &globalSum, 1, MPI_DOUBLE, 
            MPI_SUM, 0, newcomm);
    MPI_Allreduce(&maxxy, &globalMaxxy, 1,MPI_DOUBLE, 
            MPI_MAX, newcomm);
    double evenMax, oddMax, evenSum, oddSum;
    int localrank;
    MPI_Comm_rank(newcomm, &localrank);
    if (localrank == 0) {
        MPI_Sendrecv(&globalSum, 1, MPI_DOUBLE, 
                0, 0, &evenSum, 1, MPI_DOUBLE, 0, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Comm_free(&group_comm);

    MPI_Finalize();


}