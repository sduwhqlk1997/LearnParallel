set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

# Debug
#set(CMAKE_BUILD_TYPE Debug) # 生成debug符号
#add_definitions(-DDEBUG) # 定义 DEBUG 宏

find_package(MPI REQUIRED)
# find_package(OpenMP REQUIRED)
#include_directories(/home/qiu/qlk/Cppackage/Eigen/eigen-5.0.0) # jixiegeming
include_directories(/root/Cppackage/eigen/eigen-5.0.0)

# find_path(MATIO_INCLUDE_DIR matio.h)
# find_library(MATIO_LIBRARY NAMES matio)

# if(NOT MATIO_INCLUDE_DIR OR NOT MATIO_LIBRARY)
#     message(FATAL_ERROR "matio 未找到，请先安装 libmatio-dev")
# endif()

# include_directories(${MATIO_INCLUDE_DIR})

include_directories(./include)
add_executable(main main.cpp)
target_link_libraries(main PUBLIC MPI::MPI_CXX)
#target_link_libraries(main ${MATIO_LIBRARY})