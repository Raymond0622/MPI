#include <mpi.h>
#include <vector>
#include <iostream>
#include <queue>
#include <algorithm>
#include <numeric>

int main(int argc, char** argv) {
    int n = 20;
    std::vector<std::vector<int>> matrix(n, std::vector<int>(20, 0));
    for (int i = 0; i < n;i++) {
        matrix[i][i] = 2.0;
    }
    std::vector<int> vec(n), ans(n);
    for (int i = 0; i < n;i++) {
        vec[i] = 2.0;
    }
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = n / size;

    for (int i = rank * local_n;i < (rank + 1) * local_n;i++) {
        ans[i] = std::inner_product(matrix[i].begin(), matrix[i].end(), vec.begin(), 0.0);
    }

    MPI_Finalize();
        for (auto& n: ans) {
            std::cout << n << std::endl;
        }
    return 0;
}
