cmake_minimum_required(VERSION 3.10)
project(DistributedBFS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRE ON)

find_package(MPI REQUIRED)
if (MPI_FOUND) 
    message(STATUS MPI_SOURCE_DIR)
endif()

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(bfs main.cpp ${SOURCES})

target_include_directories(bfs PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(bfs PRIVATE MPI::MPI_CXX)

add_compile_definitions(GRAPH_TYPE="${GRAPH_TYPE}")

