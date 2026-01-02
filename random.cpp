#include <type_traits>
#include <iostream>
#include <stdlib.h>
#include <string>


template <size_t Row, size_t Column, typename T>
struct Matrix {
    template <size_t Row2, size_t Col2, typename T2>
    std::enable_if_t<Column == Row2, Matrix<Row, Col2, T>> operator*(const Matrix<Row2, Col2, T2>& matrix) {
        static_assert(Column == Row2);
        Matrix<Row, Col2, T> ma;
        return ma;
    }
};

void meh(const std::string& str) {
    std::cout << 1;
}

// template <typename T, typename... Ts>
// struct CountType : std::integral_constant<size_t, (std::is_same_v<T, Ts> + ...)>{};

template <typename T, typename U, typename... Ts>
struct CountType {
    static constexpr size_t value = std::is_same_v<T, U> + CountType<T, Ts...>::value;
};

template <typename T, typename U>
struct CountType<T, U> {
    static constexpr size_t value = std::is_same_v<T, U>;
};

struct Mine {
    int x;
    int y;
    std::string s;
};


int main() {
    Matrix<5, 2, int> matrix;
    Matrix<2, 5, int> matrix2;
    auto ans = matrix * matrix2;

    //std::cout << CountType<int, int, double, char, int>::value;
    meh("123");
    std::cout << std::is_trivially_copyable_v<Mine>;

}