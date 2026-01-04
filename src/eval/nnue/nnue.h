#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include <arm_neon.h>
#include <cstdint>
#include <algorithm>

namespace Nnue {

struct NNUE {
  Accumulator<1024> acc;

  WeightMatrix<1024, 64> layer1;
  ActivationVector<64> bias1;
  ActivationVector<64> hidden1;

  WeightMatrix<64, 8> layer2; 
  ActivationVector<8> bias2;
  ActivationVector<8> hidden2;

  int16_t forward() {
    // Layer 1
    multiply_crelu_optimized(&layer1, acc.accumulation, &hidden1);
    iadd(&hidden1, bias1);
  
    // Layer 2
    multiply_crelu_optimized(&layer2, hidden1, &hidden2);
    iadd(&hidden2, bias2);
  
    return hidden2.activations[0];
  }
};

}  // namespace Nnue

#endif  // SRC_EVAL_NNUE_NNUE_H