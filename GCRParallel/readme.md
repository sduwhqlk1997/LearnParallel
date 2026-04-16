# 目标：用mpi写一个并行版本的GCR算法

## 2026/04/16：**已完成** ✅  
- 并行稀疏矩阵分块  
- 并行矩阵乘向量

### 学习心得：
将all_rows数组的副本发送到所有进程:
```
MPI_Bcast(all_rows.data(), size, MPI_INT, 0, MPI_COMM_WORLD); 
```
将根进程下数组nnz_list中的值依次分给每个进程并存储在对应的my_nnz变量中:
```
MPI_Scatter(nnz_list.data(), 1, MPI_INT, &amp;my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD); 
```
将根进程下的send_vals分段发送到各个进程中，用nnz_list.data()记录每一段的长度，displs_list.data()记录每一段的起点:
```
MPI_Scatterv(send_vals.data(),
                     nnz_list.data(),
                     displs_list.data(),
                     MPI_DOUBLE,
                     recv_vals.data(),
                     my_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```                     
