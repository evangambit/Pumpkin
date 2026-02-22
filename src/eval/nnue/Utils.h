#ifndef SRC_EVAL_NNUE_UTILS_H
#define SRC_EVAL_NNUE_UTILS_H

#include "NnueFeatureBitmapType.h"

namespace NNUE {
template<typename T> struct NnueEvaluator;
}

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

#include "../../game/Position.h"

namespace NNUE {

double randn(double stddev = 1.0);

NnueFeatureBitmapType cp2nfbt(ChessEngine::ColoredPiece cp);

int16_t feature_index(NnueFeatureBitmapType feature, unsigned square);

int16_t flip_feature_index(int16_t index);

// Shift used for int16_t fixed-point scaling.
constexpr int SCALE_SHIFT = 8;
constexpr int EMBEDDING_DIM = 512;
constexpr int HIDDEN1_DIM = 64;
constexpr int OUTPUT_DIM = 1;

// 32 pieces, 4 castling rights, and 32 hanging pieces.
// We ignore tha 16 'no pawns on a file' features, since they
// can only become 1 at the expense of making other features 0.
constexpr int MAX_NUM_ONES_IN_INPUT = 32 + 4 + 32;

constexpr int16_t NNUE_INPUT_DIM = NF_COUNT * 64;

struct Features {
  uint16_t length;
  uint16_t onIndices[MAX_NUM_ONES_IN_INPUT];
  Features() : length(0) {
    std::fill_n(onIndices, MAX_NUM_ONES_IN_INPUT, NNUE_INPUT_DIM);
  }
  void addFeature(uint16_t index) {
    onIndices[length++] = static_cast<uint16_t>(index);
  }
  uint16_t operator[](size_t i) const {
    return onIndices[i];
  }
  void flip_() {
    for (size_t i = 0; i < length; i++) {
      onIndices[i] = flip_feature_index(onIndices[i]);
    }
  }
  std::vector<uint16_t> to_vector() const {
    return std::vector<uint16_t>(onIndices, onIndices + length);
  }
};

template<typename T>
Features pos2features(NnueEvaluator<T> *evaluator, const ChessEngine::Position& pos, const ChessEngine::Threats& threats) {
  Features features;
  evaluator->_evaluate(pos, threats);
  for (NnueFeatureBitmapType i = NnueFeatureBitmapType(0); i < NF_COUNT; i = NnueFeatureBitmapType(i + 1)) {
    ChessEngine::Bitboard bb = evaluator->lastPieceBitboards[i];
    while (bb) {
      unsigned sq = ChessEngine::pop_lsb_i_promise_board_is_not_empty(bb);
      features.addFeature(feature_index(i, sq));
    }
  }

  return features;
}

inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}


}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_UTILS_H