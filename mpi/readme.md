MPI class1
six basic functions
MPI_Init();
MPI_Comm_size();
MPI_Comm_rank();
MPI_Send();
MPI_Recv();
MPI_Finalize();

进程通讯形成环会产生死锁现象，解决死锁可按如下策略：
1. 偶数进程发送，奇数进程接收
2. 奇数进程发送，偶数进程接收