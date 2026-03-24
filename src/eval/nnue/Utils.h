#ifndef SRC_EVAL_NNUE_UTILS_H
#define SRC_EVAL_NNUE_UTILS_H

#include "NnueFeatureBitmapType.h"
#include "../../game/Geometry.h"

namespace NNUE {
template<typename T> struct NnueEvaluator;
}

typedef ChessEngine::File File;

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

inline int16_t feature_index(NnueFeatureBitmapType feature, unsigned square) {
  return feature * 64 + square;
}

inline ChessEngine::SafeSquare vertically_flip_square(ChessEngine::SafeSquare square) {
  return ChessEngine::SafeSquare((7 - (square / 8)) * 8 + (square % 8));
}

inline int16_t flip_feature_index(int16_t index) {
  // Flip the board position vertically (rank 8 <-> rank 1, etc.) and swap colors.
  int16_t piece_type = (((index / 64) + (NF_COUNT / 2)) % NF_COUNT) * 64;
  int16_t square = index % 64;
  int16_t flipped_square = vertically_flip_square(ChessEngine::SafeSquare(square));
  return piece_type + flipped_square;
}

// Shift used for int16_t fixed-point scaling.
constexpr int SCALE_SHIFT = 8;
constexpr int EMBEDDING_DIM = 256;
constexpr int HIDDEN1_DIM = 16;
constexpr int OUTPUT_DIM = 1;

struct Features {
  ChessEngine::SafeSquare whiteKingSquare;
  ChessEngine::SafeSquare blackKingSquare;
  std::vector<uint16_t> onIndices;
  Features() {
    onIndices.reserve(36);
  }
  Features(ChessEngine::SafeSquare whiteKingSquare, ChessEngine::SafeSquare blackKingSquare)
    : whiteKingSquare(whiteKingSquare), blackKingSquare(blackKingSquare) {
    onIndices.reserve(36);
  }
  void addFeature(uint16_t index) {
    onIndices.push_back(static_cast<uint16_t>(index));
  }
  uint16_t operator[](size_t i) const {
    return onIndices[i];
  }
  size_t size() const {
    return onIndices.size();
  }
  void flip_() {
    for (size_t i = 0; i < onIndices.size(); i++) {
      onIndices[i] = flip_feature_index(onIndices[i]);
    }
    ChessEngine::SafeSquare temp = whiteKingSquare;
    whiteKingSquare = vertically_flip_square(blackKingSquare);
    blackKingSquare = vertically_flip_square(temp);
  }
  std::vector<uint16_t> to_vector() const {
    return onIndices;
  }
};

inline ChessEngine::Bitboard nnue_feature_to_bitboard(NnueFeatureBitmapType feature, const ChessEngine::Position& pos, const ChessEngine::Threats& threats) {
  switch (feature) {
    case NF_WHITE_PAWN: {
      // We use the 0th rank (0 - 7) to store castling rights. This
      // trick works because pawns can never occupy these squares,
      // so these bits are unused. Importantly, everything is
      // vertically flipped for the black pawns (i.e. castling rights
      // use the 7th rank). This way the same vertical symmetry that we
      // use for our piece features automatically works for these features too.
      ChessEngine::Bitboard r = pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_PAWN];
      if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_WhiteKing) {
        r |= ChessEngine::bb(0);
      }
      if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_WhiteQueen) {
        r |= ChessEngine::bb(1);
      }
      return r;
    }
    case NF_WHITE_KNIGHT:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_KNIGHT];
    case NF_WHITE_BISHOP:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_BISHOP];
    case NF_WHITE_ROOK:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_ROOK];
    case NF_WHITE_QUEEN:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_QUEEN];
    case NF_BLACK_PAWN: {
      ChessEngine::Bitboard r = pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_PAWN];
      if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_BlackKing) {
        r |= ChessEngine::bb(56);
      }
      if (pos.currentState_.castlingRights & ChessEngine::kCastlingRights_BlackQueen) {
        r |= ChessEngine::bb(57);
      }
      return r;
    }
    case NF_BLACK_KNIGHT:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_KNIGHT];
    case NF_BLACK_BISHOP:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_BISHOP];
    case NF_BLACK_ROOK:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_ROOK];
    case NF_BLACK_QUEEN:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_QUEEN];
    default:
      std::cerr << "Invalid NnueFeatureBitmapType: " << feature << std::endl;
  }
  return ChessEngine::kEmptyBitboard;
}

inline Features pos2features(const ChessEngine::Position& pos, const ChessEngine::Threats& threats) {
  ChessEngine::SafeSquare whiteKingSquare = ChessEngine::lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_KING]);
  ChessEngine::SafeSquare blackKingSquare = ChessEngine::lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_KING]);
  Features features(whiteKingSquare, blackKingSquare);
  for (NnueFeatureBitmapType i = NnueFeatureBitmapType(0); i < NF_COUNT; i = NnueFeatureBitmapType(i + 1)) {
    ChessEngine::Bitboard bb = nnue_feature_to_bitboard(i, pos, threats);
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