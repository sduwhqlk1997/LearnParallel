#include <iostream>
#include<unistd.h>
#include "mpi.h"

int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    sleep(100);
    MPI_Finalize();
    return 0;
}