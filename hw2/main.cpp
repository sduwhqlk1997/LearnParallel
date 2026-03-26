#include <iostream>
#include <string>
#include <thread>

void say_hi(int num){
    std::cout<<"Hi"+std::to_string(num)+"\n";
}

int main(){
    std::thread hi_thread0(say_hi,0);
    std::thread hi_thread1(say_hi,1);
    std::thread hi_thread2(say_hi,2);
    std::thread hi_thread3(say_hi,3);
    hi_thread0.join();
    hi_thread1.join();
    hi_thread2.join();
    hi_thread3.join();
    return 0;
}