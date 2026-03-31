#include <iostream>
#include <future>
void say_hi(){
    std::cout<<"Hello OpenFOAM"<<std::endl;
}
int main(){
    auto yang = std::async(say_hi);
    return 0;
}