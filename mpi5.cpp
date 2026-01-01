#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <vector>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int local_size = rank + 1;
    int tot = 10;
    std::vector<int> vec(local_size), counts(1), displs(1), global(2);
    for (int i = 0; i < local_size;i++) {
        vec[i] = rank* 10 + i;
    }
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);

        int offset = 0;
        for (int i = 0; i < size; i++) {
            counts[i] = i + 1;
            displs[i] = offset;
            offset += counts[i];
        }
        global.resize(offset);
    }
    MPI_Gatherv(vec.data(), local_size, MPI_INT, 
        global.data(), counts.data(), displs.data(), MPI_INT, 0, 
            MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Gathered data: ";
        for (int v : global) std::cout << v << " ";
        std::cout << std::endl;
    }
    MPI_Finalize();
}