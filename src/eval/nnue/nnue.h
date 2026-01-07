#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include <eigen3/Eigen/Dense>
#include <cstdint>
#include <algorithm>

#include "../../game/Position.h"

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
    spare = v * s * stddev;
    return u * s * stddev;
}


constexpr int SCALE_SHIFT = 6;
constexpr int EMBEDDING_DIM = 1024;
constexpr int HIDDEN1_DIM = 128;
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

int16_t feature_index(ChessEngine::SafeColoredPiece piece, unsigned square) {
  return piece * 64 + square;
}

int16_t flip_feature_index(int16_t index) {
  return index;  // TODO
}

Features pos2features(const struct ChessEngine::Position& pos) {
  Features features;
  for (unsigned i = 0; i < 64; ++i) {
    ChessEngine::SafeSquare sq = ChessEngine::SafeSquare(i);
    ChessEngine::ColoredPiece piece = pos.tiles_[sq];
    if (piece != ChessEngine::ColoredPiece::NO_COLORED_PIECE) {
      features.addFeature(feature_index(to_safe_colored_piece(piece), sq));
    }
  }
  if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_WhiteKing) {
    features.addFeature(SpecialFeatures::WHITE_KINGSIDE_CASTLING_RIGHT);
  }
  if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_WhiteQueen) {
    features.addFeature(SpecialFeatures::WHITE_QUEENSIDE_CASTLING_RIGHT);
  }
  if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_BlackKing) {
    features.addFeature(SpecialFeatures::BLACK_KINGSIDE_CASTLING_RIGHT);
  }
  if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_BlackQueen) {
    features.addFeature(SpecialFeatures::BLACK_QUEENSIDE_CASTLING_RIGHT);
  }
  return features;
}

struct Nnue {
  bool x[INPUT_DIM];
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> embWeights[INPUT_DIM];
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> whiteAcc;
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> blackAcc;

  Eigen::Matrix<int16_t, Eigen::Dynamic, HIDDEN1_DIM> layer1;
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
    whiteAcc += embWeights[index];
    blackAcc += embWeights[flip_feature_index(index)];
  }

  void decrement(size_t index) {
    assert(x[index]);
    x[index] = false;
    whiteAcc -= embWeights[index];
    blackAcc -= embWeights[flip_feature_index(index)];
  }

  void clear_accumulator() {
    whiteAcc.setZero();
    blackAcc.setZero();
  }

  void zero_() {
    whiteAcc.setZero();
    blackAcc.setZero();
    layer1.setZero(EMBEDDING_DIM, HIDDEN1_DIM);
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
      embWeights[i].array() = Eigen::Array<int16_t, 1, EMBEDDING_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / MAX_NUM_ONES_IN_INPUT)) * (1 << SCALE_SHIFT)); });
    }
    layer1.array() = Eigen::Array<int16_t, EMBEDDING_DIM, HIDDEN1_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / EMBEDDING_DIM)) * (1 << SCALE_SHIFT)); });
    bias1.array() = Eigen::Array<int16_t, 1, HIDDEN1_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
    layer2.array() = Eigen::Array<int16_t, HIDDEN1_DIM, OUTPUT_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / HIDDEN1_DIM)) * (1 << SCALE_SHIFT)); });
    bias2.array() = Eigen::Array<int16_t, 1, OUTPUT_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
    this->compute_acc_from_scratch();
  }

  void compute_acc_from_scratch() {
    whiteAcc.setZero();
    blackAcc.setZero();
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      if (x[i]) {
        whiteAcc += embWeights[i];
        blackAcc += embWeights[flip_feature_index(i)];
      }
    }
  }

  int16_t *forward() {
    // TODO: use blackAcc too.
    hidden1.array() = (whiteAcc * layer1 + bias1).array().cwiseMax(0).cwiseMin(127); 
    hidden2.array() = (hidden1 * layer2 + bias2).array();
    return hidden2.data();
  }
};

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUE_H