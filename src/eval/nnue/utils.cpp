#include "utils.h"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace NNUE {

double randn(double stddev) {
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

int16_t feature_index(ChessEngine::SafeColoredPiece piece, unsigned square) {
  return piece * 64 + square;
}

int16_t flip_feature_index(int16_t index) {
  // Flip the board position vertically (rank 8 <-> rank 1, etc.) and swap colors.
  int16_t piece_type = (((index / 64) + 6) % 12) * 64;
  int16_t square = index % 64;
  int16_t flipped_square = (7 - (square / 8)) * 8 + (square % 8);
  return piece_type + flipped_square;
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


}