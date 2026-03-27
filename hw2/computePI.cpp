#include<mutex>
#include <iostream>
#include <thread>
#include <vector>
double Pi=0.0;
std::mutex pi_mutex;
void numInt(const double a, const double b, const double h){
    double parrallel_pi=0.0;
    for(double i=a;i<b;i+=h){
        parrallel_pi+=4*h/(1+i*i);
    }
    { // lock
        std::lock_guard<std::mutex> pi_guard(pi_mutex);
        Pi+=parrallel_pi;
    }
}

int main(){
    std::vector<std::thread> Mythread;
    Mythread.reserve(4);
    size_t N=1024;
    double h = 1/static_cast<double>(N);
    for(int i=0;i<4;i++){
        Mythread.emplace_back(std::thread(numInt,i*0.25,(i+1)*0.25,h));
    }
    for(int i=0;i<4;i++){
        Mythread[i].join();
    }
    std::cout<<"Pi is : "<<Pi<<std::endl;
    return 0;
}