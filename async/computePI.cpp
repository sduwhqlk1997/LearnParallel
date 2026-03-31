#include<iostream>
#include<future>
#include<iomanip>
#include<vector>

double pi_helper(int begin, int end, double h){
    double partial_pi=0.0;
    for(int i=begin;i<end;++i){
        partial_pi +=(h*4.0)/(1+i*h*i*h);
    }
    return partial_pi;
}

int main(int argc, char* argv[]){
    using std::chrono::steady_clock;
    using std::chrono::milliseconds;
    using std::chrono::duration_cast;
    size_t interavls=1024*1024*1024;
    size_t partitions = 2;
    if (argc >=2) partitions=std::stoul(argv[1]); // argv[1]时程序运行时手动输入的第一个参数

    size_t blocksize=interavls/partitions;
    double h = 1/static_cast<double>(interavls);

    auto start{steady_clock::now()};
    std::vector<std::future<double>> futs;
    for(size_t i=0;i<partitions;++i){
        futs.push_back(std::async(std::launch::async,pi_helper,i*blocksize,(i+1)*blocksize,h));
    }
    double pi=0.0;
    for(size_t i=0; i<partitions;++i){
        pi+=futs[i].get();
    }
    auto end{steady_clock::now()};
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"PI is "<<pi<<"\tTime is: "<<std::fixed<<std::setprecision(15)<<time/1000<<std::endl;
    return 0;
}