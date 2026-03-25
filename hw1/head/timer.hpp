#ifndef __TIMER_HPP__
#define __TIMER_HPP__
#include <chrono>
#include <ratio>
class Timer
{
public:
    using time_t=std::chrono::time_point<std::chrono::steady_clock>;
    Timer():start_time(), end_time(){};
    ~Timer(){}
    // time_t start(){return start_time=std::chrono::steady_clock::now();}
    time_t start() { return start_time = std::chrono::steady_clock::now(); }
    time_t end() {return end_time = std::chrono::steady_clock::now();}
    double duration(){return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count();}
private:
    time_t start_time, end_time;
};
#endif
