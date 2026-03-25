#include<iostream>
#include "matrix.hpp"
#include "timer.hpp"
using namespace std;
int main(){
    cout<< "N\tduration"<<endl;

    for(int N=8; N<512; N *=2){
        Matrix A(N,N,true), B(N,N), C(N,N);
        Timer T; T.start();
        A=B*C;
        T.end();

        cout<<N<<"\t"<<T.duration()<<endl;
    }
    return 0;
}