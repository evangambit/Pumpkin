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
constexpr int EMBEDDING_DIM = 1024;
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

inline ChessEngine::Bitboard nnue_feature_to_bitboard(NnueFeatureBitmapType feature, const ChessEngine::Position& pos, const ChessEngine::Threats& threats) {
  switch (feature) {
    case NF_WHITE_PAWN: {
      // We use the 7th rank (56 - 63) to store whether there are any
      // white pawns on a file. We use the 0th rank (0 - 7) to store
      // castling rights. This trick works because pawns can never
      // occupy these squares, so these bits are unused. Importantly,
      // everything is vertically flipped for the black pawns (i.e.
      // open files use the 0th rank and castling rights use the 7th
      // rank). This way the same vertical symmetry that we use for
      // our piece features automatically works for these features too.
      ChessEngine::Bitboard r = pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_PAWN];
      for (int file = 0; file < 8; file++) {
        const bool noWhitePawnsOnFile = (ChessEngine::kFiles[file] & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_PAWN]) == ChessEngine::kEmptyBitboard;
        r |= ChessEngine::bb(56 + file);
      }
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
    case NF_WHITE_KING:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_KING];
    case NF_WHITE_HANGING_PAWNS: 
      return threats.badForCp(ChessEngine::ColoredPiece::WHITE_PAWN) & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_PAWN];
    case NF_WHITE_HANGING_KNIGHTS:
      return threats.badForCp(ChessEngine::ColoredPiece::WHITE_KNIGHT) & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_KNIGHT];
    case NF_WHITE_HANGING_BISHOPS:
      return threats.badForCp(ChessEngine::ColoredPiece::WHITE_BISHOP) & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_BISHOP];
    case NF_WHITE_HANGING_ROOKS:
      return threats.badForCp(ChessEngine::ColoredPiece::WHITE_ROOK) & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_ROOK];
    case NF_WHITE_HANGING_QUEENS:
      return threats.badForCp(ChessEngine::ColoredPiece::WHITE_QUEEN) & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_QUEEN];
    case NF_WHITE_HANGING_KINGS:
      return threats.badForCp(ChessEngine::ColoredPiece::WHITE_KING) & pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_KING];
    case NF_BLACK_PAWN: {
      ChessEngine::Bitboard r = pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_PAWN];
      for (int file = 0; file < 8; file++) {
        const bool noBlackPawnsOnFile = (ChessEngine::kFiles[file] & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_PAWN]) == ChessEngine::kEmptyBitboard;
        r |= ChessEngine::bb(file);
      }
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
    case NF_BLACK_KING:
      return pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_KING];
    case NF_BLACK_HANGING_PAWNS:
      return threats.badForCp(ChessEngine::ColoredPiece::BLACK_PAWN) & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_PAWN];
    case NF_BLACK_HANGING_KNIGHTS:
      return threats.badForCp(ChessEngine::ColoredPiece::BLACK_KNIGHT) & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_KNIGHT];
    case NF_BLACK_HANGING_BISHOPS:
      return threats.badForCp(ChessEngine::ColoredPiece::BLACK_BISHOP) & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_BISHOP];
    case NF_BLACK_HANGING_ROOKS:
      return threats.badForCp(ChessEngine::ColoredPiece::BLACK_ROOK) & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_ROOK];
    case NF_BLACK_HANGING_QUEENS:
      return threats.badForCp(ChessEngine::ColoredPiece::BLACK_QUEEN) & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_QUEEN];
    case NF_BLACK_HANGING_KINGS:
      return threats.badForCp(ChessEngine::ColoredPiece::BLACK_KING) & pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_KING];
    default:
      std::cerr << "Invalid NnueFeatureBitmapType: " << feature << std::endl;
  }
  return ChessEngine::kEmptyBitboard;
}

template<typename T>
Features pos2features(NnueEvaluator<T> *evaluator, const ChessEngine::Position& pos, const ChessEngine::Threats& threats) {
  Features features;
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