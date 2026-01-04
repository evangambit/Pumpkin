#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include <eigen3/Eigen/Dense>
#include <cstdint>
#include <algorithm>

namespace NNUE {

double randn(double stddev = 1.0) {
    // Box-Muller transform
    static bool hasSpare = false;
    static double spare;

    if (hasSpare) {
        hasSpare = false;
        return spare;
    }

    hasSpare = true;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    return u * s * stddev;
}


constexpr int SCALE_SHIFT = 6;
constexpr int INPUT_DIM = 768;
constexpr int EMBEDDING_DIM = 1024;
constexpr int HIDDEN1_DIM = 64;
constexpr int OUTPUT_DIM = 8;

struct Nnue {
  bool x[INPUT_DIM];
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> embWeights[INPUT_DIM];
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> acc;

  Eigen::Matrix<int16_t, EMBEDDING_DIM, HIDDEN1_DIM> layer1;
  Eigen::Matrix<int16_t, 1, HIDDEN1_DIM> bias1;
  Eigen::Matrix<int16_t, 1, HIDDEN1_DIM> hidden1;

  Eigen::Matrix<int16_t, HIDDEN1_DIM, OUTPUT_DIM> layer2;
  Eigen::Matrix<int16_t, 1, OUTPUT_DIM> bias2;
  Eigen::Matrix<int16_t, 1, OUTPUT_DIM> hidden2;

  Nnue() {
    std::fill_n(x, INPUT_DIM, false);
    zero_();
  }

  void increment(size_t index) {
    assert(!x[index]);
    x[index] = true;
    acc += embWeights[index];
  }

  void decrement(size_t index) {
    assert(x[index]);
    x[index] = false;
    acc -= embWeights[index];
  }

  void zero_() {
    acc.setZero();
    layer1.setZero();
    bias1.setZero();
    hidden1.setZero();
    layer2.setZero();
    bias2.setZero();
    hidden2.setZero();
  }

  void randn_() {
    /**
      Initialize weights and biases with gaussian random values.
     */
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      embWeights[i].array() = Eigen::Array<int16_t, 1, EMBEDDING_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(1.0 / EMBEDDING_DIM) * (1 << SCALE_SHIFT)); });
    }
    layer1.array() = Eigen::Array<int16_t, EMBEDDING_DIM, HIDDEN1_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(1.0 / HIDDEN1_DIM) * (1 << SCALE_SHIFT)); });
    bias1.array() = Eigen::Array<int16_t, 1, HIDDEN1_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
    layer2.array() = Eigen::Array<int16_t, HIDDEN1_DIM, OUTPUT_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(1.0 / OUTPUT_DIM) * (1 << SCALE_SHIFT)); });
    bias2.array() = Eigen::Array<int16_t, 1, OUTPUT_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
    this->compute_acc_from_scratch();
  }

  void compute_acc_from_scratch() {
    acc.setZero();
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      if (x[i]) {
        acc += embWeights[i];
      }
    }
  }

  int16_t *forward() {
    // Layer 1
    hidden1.array() = (acc * layer1 + bias1).array().cwiseMax(0).cwiseMin(127); 
    // Layer 2
    hidden2.array() = (hidden1 * layer2 + bias2).array();
  
    return hidden2.data();
  }
};

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUE_H