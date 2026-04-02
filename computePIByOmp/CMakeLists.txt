set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenMP REQUIRED)
add_executable(computePI main.cpp)
target_link_libraries(computePI PUBLIC OpenMP::OpenMP_CXX)