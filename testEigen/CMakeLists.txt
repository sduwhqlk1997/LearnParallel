set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenMP REQUIRED)
include_directories(/home/qiu/qlk/Cppackage/Eigen/eigen-5.0.0)
add_executable(main main.cpp)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)