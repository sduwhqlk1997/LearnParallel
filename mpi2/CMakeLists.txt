set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_BUILD_TYPE RELEASE)
find_package(MPI REQUIRED)

add_executable(main Possion1D.cpp)

target_link_libraries(main PUBLIC MPI::MPI_CXX)