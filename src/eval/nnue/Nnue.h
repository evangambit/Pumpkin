#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include "NnueFeatureBitmapType.h"
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../../game/Position.h"
#include "Matrix.h"
#include "Utils.h"

namespace NNUE {

template<typename T>
struct Nnue {
  Vector<EMBEDDING_DIM, T> embWeights[MAX_FEATURE_INDEX];

  Matrix<HIDDEN1_DIM, EMBEDDING_DIM * 2, int8_t> layer1;
  Vector<HIDDEN1_DIM, T> bias1;
  Vector<HIDDEN1_DIM, T> hidden1;

  Matrix<OUTPUT_DIM, HIDDEN1_DIM, T> layer2;
  Vector<OUTPUT_DIM, T> bias2;
  Vector<OUTPUT_DIM, T> output;

  Vector<EMBEDDING_DIM, T> clippedMover;
  Vector<EMBEDDING_DIM, T> clippedOpponent;

  Nnue() {
    layer1.setZero();
    bias1.setZero();
    layer2.setZero();
    bias2.setZero();
    output.setZero();
  }

  void increment(Vector<EMBEDDING_DIM, T> *whiteAcc, size_t whiteIndex,
                  Vector<EMBEDDING_DIM, T> *blackAcc, size_t blackIndex) {
    *whiteAcc += embWeights[whiteIndex];
    *blackAcc += embWeights[blackIndex];
  }

  void decrement(Vector<EMBEDDING_DIM, T> *whiteAcc, size_t whiteIndex,
                  Vector<EMBEDDING_DIM, T> *blackAcc, size_t blackIndex) {
    *whiteAcc -= embWeights[whiteIndex];
    *blackAcc -= embWeights[blackIndex];
  }

  void randn_() {
    for (size_t i = 0; i < MAX_FEATURE_INDEX; ++i) {
      embWeights[i].randn_();
    }
    layer1.randn_();
    bias1.randn_();
    layer2.randn_();
    bias2.randn_();
  }

  void load(std::istream& in) {
    auto emb = std::make_unique<Matrix<MAX_FEATURE_INDEX, EMBEDDING_DIM, T>>();
    emb->load_from_stream(in);
    for (size_t i = 0; i < MAX_FEATURE_INDEX; ++i) {
      embWeights[i].load_from_row(*emb, i);
    }
    layer1.load_from_stream(in);
    bias1.load_from_stream(in);
    layer2.load_from_stream(in);
    bias2.load_from_stream(in);

    // Verify that the entire file has been read
    char dummy;
    if (in.read(&dummy, 1) || !in.eof()) {
      throw std::runtime_error("File not completely read");
    }
  }

  void use_debug_weights() {
    for (size_t i = 0; i < MAX_FEATURE_INDEX; ++i) {
      embWeights[i].setZero();
      if (i < EMBEDDING_DIM) {
        embWeights[i].data[i] = static_cast<int16_t>(1);
      }
    }
    layer1.randn_();
    bias1.randn_();
    layer2.randn_();
    bias2.randn_();
  }

  T *forward(const Vector<EMBEDDING_DIM, T>& mover, const Vector<EMBEDDING_DIM, T>& opponent) {
    this->clippedMover = mover;
    this->clippedOpponent = opponent;
    
    constexpr T maxValue = std::is_same<T, float>::value ? T(1) : T(1 << SCALE_SHIFT);
    this->clippedMover.clip_(T(0), maxValue);
    this->clippedOpponent.clip_(T(0), maxValue);
    
    if constexpr (std::is_same<T, int16_t>::value) {
      Vector<EMBEDDING_DIM, int8_t> qmover;
      Vector<EMBEDDING_DIM, int8_t> qopponent;
#if defined(__ARM_NEON) || defined(__aarch64__)
      for (size_t j = 0; j < EMBEDDING_DIM; j += 16) {
        int16x8_t m0 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedMover.data[j]));
        int16x8_t m1 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedMover.data[j + 8]));
        int8x8_t mq0 = vqshrn_n_s16(m0, 2);
        int8x8_t mq1 = vqshrn_n_s16(m1, 2);
        vst1q_s8(&qmover.data[j], vcombine_s8(mq0, mq1));

        int16x8_t o0 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedOpponent.data[j]));
        int16x8_t o1 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedOpponent.data[j + 8]));
        int8x8_t oq0 = vqshrn_n_s16(o0, 2);
        int8x8_t oq1 = vqshrn_n_s16(o1, 2);
        vst1q_s8(&qopponent.data[j], vcombine_s8(oq0, oq1));
      }
#elif defined(__AVX2__)
      for (size_t j = 0; j < EMBEDDING_DIM; j += 32) {
        __m256i m0 = _mm256_load_si256((const __m256i*)&clippedMover.data[j]);
        __m256i m1 = _mm256_load_si256((const __m256i*)&clippedMover.data[j + 16]);
        
        m0 = _mm256_srai_epi16(m0, 2);
        m1 = _mm256_srai_epi16(m1, 2);
        
        __m256i packed = _mm256_packs_epi16(m0, m1);
        __m256i ordered = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_store_si256((__m256i*)&qmover.data[j], ordered);

        __m256i o0 = _mm256_load_si256((const __m256i*)&clippedOpponent.data[j]);
        __m256i o1 = _mm256_load_si256((const __m256i*)&clippedOpponent.data[j + 16]);
        o0 = _mm256_srai_epi16(o0, 2);
        o1 = _mm256_srai_epi16(o1, 2);
        __m256i opacked = _mm256_packs_epi16(o0, o1);
        __m256i oordered = _mm256_permute4x64_epi64(opacked, _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_store_si256((__m256i*)&qopponent.data[j], oordered);
      }
#else
      for (size_t j = 0; j < EMBEDDING_DIM; ++j) {
        qmover.data[j] = std::max<int16_t>(-(1 << 7), std::min<int16_t>((1 << 7) - 1, clippedMover.data[j] >> 2));
        qopponent.data[j] = std::max<int16_t>(-(1 << 7), std::min<int16_t>((1 << 7) - 1, clippedOpponent.data[j] >> 2));
      }
#endif
      concat_and_matmul_int8<HIDDEN1_DIM, EMBEDDING_DIM * 2, EMBEDDING_DIM, EMBEDDING_DIM>(
          layer1, qmover, qopponent, reinterpret_cast<Vector<HIDDEN1_DIM, int16_t>*>(&hidden1));
    } else {
      // Float implementation is for debugging. It is not intended to be highly optimized.
      concat_and_matmul<HIDDEN1_DIM, EMBEDDING_DIM * 2, EMBEDDING_DIM, EMBEDDING_DIM>(
          layer1, clippedMover, clippedOpponent, reinterpret_cast<Vector<HIDDEN1_DIM, float>*>(&hidden1));
    }
  
    hidden1 += bias1;
    hidden1.clip_(T(0), maxValue);

    matmul(layer2, hidden1, &output);
    this->output += bias2;
    return this->output.data_ptr();
  }

  std::shared_ptr<Nnue> clone() const {
    std::shared_ptr<Nnue> copy = std::make_shared<Nnue>();
    for (size_t i = 0; i < MAX_FEATURE_INDEX; ++i) {
      copy->embWeights[i] = this->embWeights[i];
    }
    copy->layer1 = this->layer1;
    copy->bias1 = this->bias1;
    copy->layer2 = this->layer2;
    copy->bias2 = this->bias2;
    return copy;
  }
};

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUE_H
