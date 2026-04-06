set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenMP REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)