#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <vector>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<int> vec2(4 / size);
    std::vector<int> vec(4);
    int local = 0, global = 0;
    if (rank == 0) {
        
        for (int i = 0; i < vec.size();i++) {
            vec[i] = i + 1;
        } 
         
    }
    MPI_Scatter(vec.data(), vec2.size(), MPI_INT, 
            vec2.data(), vec2.size(), MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < vec2.size();i++) {
        local += vec2[i] * vec2[i];
    }
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " : " << std::sqrt(global) << std::endl;
    MPI_Finalize();
}
