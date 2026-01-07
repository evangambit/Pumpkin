#include <iostream>
#include <cstdint>

#include <eigen3/Eigen/Dense>

#include "nnue.h"

using namespace NNUE;

// g++ src/eval/nnue/main.cpp --std=c++20 -I/usr/local/include -L/usr/local/lib && ./a.out

template<class T>
void print8(const T* data) {
    for (int i = 0; i < 8; ++i) {
        std::cout << float(data[i]) / float(1 << SCALE_SHIFT) << " ";
    }
    std::cout << std::endl;
}

int main(int, char**) {
    Nnue nnue;

    nnue.randn_();
    nnue.increment(0);
    nnue.forward();

    print8(nnue.embWeights[0].data());

    std::cout << "acc: ";
    print8(nnue.acc.data());

    std::cout << "hidden1: ";
    print8(nnue.hidden1.data());

    std::cout << "hidden2: ";
    print8(nnue.hidden2.data());


    std::cout << "output: ";
    print8(nnue.forward());

    return 0;
}
