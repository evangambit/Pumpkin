#include <arm_neon.h>
#include <cstdint>
#include <algorithm>

constexpr int kInputDim = 768;
const int SCALE_SHIFT = 6;

template <int D>
struct alignas(16) ActivationVector {
  static_assert(D % 16 == 0, "D must be a multiple of 16 for peak SIMD throughput");
  int16_t activations[D];
};

template <int H, int W>
struct alignas(64) WeightMatrix {
  static_assert(H % 8 == 0 && W % 16 == 0, "H must be multiple of 8, W multiple of 16");
  int8_t weights[W * H]; 
};

template <int D>
void iadd(ActivationVector<D>& *acc, const ActivationVector<D>& w) {
  int16_t* __restrict a_ptr = acc->activations;
  const int16_t* __restrict w_ptr = w.activations;
  for (int i = 0; i < D; i += 16) {
    int16x8_t l = vld1q_s16(a_ptr + i);
    int16x8_t h = vld1q_s16(a_ptr + i + 8);
    vst1q_s16(a_ptr + i, vaddq_s16(l, vld1q_s16(w_ptr + i)));
    vst1q_s16(a_ptr + i + 8, vaddq_s16(h, vld1q_s16(w_ptr + i + 8)));
  }
}

template <int D>
void isub(ActivationVector<D>& *acc, const ActivationVector<D>& w) {
  int16_t* __restrict a_ptr = acc->activations;
  const int16_t* __restrict w_ptr = w.activations;
  for (int i = 0; i < D; i += 16) {
    int16x8_t l = vld1q_s16(a_ptr + i);
    int16x8_t h = vld1q_s16(a_ptr + i + 8);
    vst1q_s16(a_ptr + i, vsubq_s16(l, vld1q_s16(w_ptr + i)));
    vst1q_s16(a_ptr + i + 8, vsubq_s16(h, vld1q_s16(w_ptr + i + 8)));
  }
}

// Dense Fully Connected Layer with CReLU
template <int H, int W>
void multiply_crelu(const WeightMatrix<H, W>& matrix, const ActivationVector<H>& input,
          ActivationVector<W> *output) {
  for (int w = 0; w < W; w += 4) {
    int32x4_t acc0_lo = vdupq_n_s32(0), acc0_hi = vdupq_n_s32(0);
    int32x4_t acc1_lo = vdupq_n_s32(0), acc1_hi = vdupq_n_s32(0);
    int32x4_t acc2_lo = vdupq_n_s32(0), acc2_hi = vdupq_n_s32(0);
    int32x4_t acc3_lo = vdupq_n_s32(0), acc3_hi = vdupq_n_s32(0);

    for (int h = 0; h < H; h += 8) {
      int16x8_t a = vld1q_s16(&input.activations[h]);
      int16x4_t a_lo = vget_low_s16(a), a_hi = vget_high_s16(a);

      auto process_row = [&](int offset, int32x4_t& lo, int32x4_t& hi) {
        int16x8_t w_vec = vmovl_s8(vld1_s8(&matrix.weights[offset + h]));
        lo = vmlal_s16(lo, vget_low_s16(w_vec), a_lo);
        hi = vmlal_s16(hi, vget_high_s16(w_vec), a_hi);
      };

      process_row((w + 0) * H, acc0_lo, acc0_hi);
      process_row((w + 1) * H, acc1_lo, acc1_hi);
      process_row((w + 2) * H, acc2_lo, acc2_hi);
      process_row((w + 3) * H, acc3_lo, acc3_hi);
    }

    auto finalize = [](int32x4_t lo, int32x4_t hi) {
      int32_t sum = vaddvq_s32(vaddq_s32(lo, hi)) >> SCALE_SHIFT;
      return static_cast<int16_t>(std::max(0, std::min(127, (int)sum)));
    };

    output->activations[w+0] = finalize(acc0_lo, acc0_hi);
    output->activations[w+1] = finalize(acc1_lo, acc1_hi);
    output->activations[w+2] = finalize(acc2_lo, acc2_hi);
    output->activations[w+3] = finalize(acc3_lo, acc3_hi);
  }
}

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
