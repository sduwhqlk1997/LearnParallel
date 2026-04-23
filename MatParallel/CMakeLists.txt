set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(/home/qiu/qlk/Cppackage/Eigen/eigen-5.0.0) # jixiegeming
# include_directories(/root/Cppackage/eigen/eigen-5.0.0)

include_directories(./include)
add_executable(main main2.cpp)
target_link_libraries(main PUBLIC MPI::MPI_CXX)