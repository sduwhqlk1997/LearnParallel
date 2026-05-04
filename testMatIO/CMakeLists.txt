cmake_minimum_required(VERSION 3.10)
project(ReadMatSparseComplex)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 包含 Eigen 头文件
# 如果你把 Eigen 解压到了指定目录，就用下面这句（替换成你的 Eigen 路径）
include_directories(/home/qiu/qlk/Cppackage/Eigen/eigen-5.0.0)  # Ubuntu 默认路径
# include_directories(D:/eigen-3.4.0)     # Windows 示例路径

# 2. 查找 matio 库
# find_library(MATIO_LIB matio REQUIRED)
set(MATIO_LIB /home/qiu/qlk/Cppackage/matio/lib/libmatio.so)
include_directories(/home/qiu/qlk/Cppackage/matio/include)  # matio 头文件路径

# 生成可执行文件
add_executable(read_mat main.cpp)

# 链接 matio 库
target_link_libraries(read_mat ${MATIO_LIB})