#ifndef SRC_EVAL_NNUE_UTILS_H
#define SRC_EVAL_NNUE_UTILS_H

#include <eigen3/Eigen/Dense>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "../../game/Position.h"

namespace NNUE {

double randn(double stddev = 1.0);

constexpr int SCALE_SHIFT = 8;
constexpr int EMBEDDING_DIM = 1024;
constexpr int HIDDEN1_DIM = 256;
constexpr int HIDDEN2_DIM = 64;
constexpr int OUTPUT_DIM = 16;

constexpr int MAX_NUM_ONES_IN_INPUT = 32 + 4;

enum SpecialFeatures : int16_t {
  // It is impossible for a pawn to be on the first or last rank, so we can
  // use these indices to encode other things.

  // We use some unused pawn positions to encode castling rights.
  // We choose squares so that the same flipping logic that applies to pieces
  // also applies to castling rights.
  WHITE_KINGSIDE_CASTLING_RIGHT = 0,  // "white pawn on a8"
  WHITE_QUEENSIDE_CASTLING_RIGHT = 1,  // "white pawn on b8"
  BLACK_KINGSIDE_CASTLING_RIGHT = 440,  // "black pawn on a1" (vertically flipped vs white's castling right)
  BLACK_QUEENSIDE_CASTLING_RIGHT = 441,  // "black pawn on b1" (vertically flipped vs white's castling right)

  WHITE_PAWN_ON_A8 = 0,
  WHITE_KNIGHT_ON_A8 = 64,
  WHITE_BISHOP_ON_A8 = 128,
  WHITE_ROOK_ON_A8 = 192,
  WHITE_QUEEN_ON_A8 = 256,
  WHITE_KING_ON_A8 = 320,
  BLACK_PAWN_ON_A8 = 384,
  BLACK_KNIGHT_ON_A8 = 448,
  BLACK_BISHOP_ON_A8 = 512,
  BLACK_ROOK_ON_A8 = 576,
  BLACK_QUEEN_ON_A8 = 640,
  BLACK_KING_ON_A8 = 704,
  INPUT_DIM = 768,
};

struct Features {
  uint16_t length;
  int16_t onIndices[MAX_NUM_ONES_IN_INPUT];
  Features() : length(0) {
    std::fill_n(onIndices, MAX_NUM_ONES_IN_INPUT, SpecialFeatures::INPUT_DIM);
  }
  void addFeature(uint16_t index) {
    onIndices[length++] = static_cast<uint16_t>(index);
  }
  uint16_t operator[](size_t i) const {
    return onIndices[i];
  }
};

int16_t feature_index(ChessEngine::SafeColoredPiece piece, unsigned square);

int16_t flip_feature_index(int16_t index);

Features pos2features(const struct ChessEngine::Position& pos);

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_UTILS_H