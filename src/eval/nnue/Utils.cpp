#include "Utils.h"

#include "../../game/Threats.h"
#include "../../game/CreateThreats.h"

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

NnueFeatureBitmapType cp2nfbt(ChessEngine::ColoredPiece cp) {
  switch (cp) {
    case ChessEngine::ColoredPiece::WHITE_PAWN: return NF_WHITE_PAWN;
    case ChessEngine::ColoredPiece::WHITE_KNIGHT: return NF_WHITE_KNIGHT;
    case ChessEngine::ColoredPiece::WHITE_BISHOP: return NF_WHITE_BISHOP;
    case ChessEngine::ColoredPiece::WHITE_ROOK: return NF_WHITE_ROOK;
    case ChessEngine::ColoredPiece::WHITE_QUEEN: return NF_WHITE_QUEEN;
    case ChessEngine::ColoredPiece::WHITE_KING: return NF_WHITE_KING;
    case ChessEngine::ColoredPiece::BLACK_PAWN: return NF_BLACK_PAWN;
    case ChessEngine::ColoredPiece::BLACK_KNIGHT: return NF_BLACK_KNIGHT;
    case ChessEngine::ColoredPiece::BLACK_BISHOP: return NF_BLACK_BISHOP;
    case ChessEngine::ColoredPiece::BLACK_ROOK: return NF_BLACK_ROOK;
    case ChessEngine::ColoredPiece::BLACK_QUEEN: return NF_BLACK_QUEEN;
    case ChessEngine::ColoredPiece::BLACK_KING: return NF_BLACK_KING;
    default: {
      throw std::invalid_argument("Invalid ColoredPiece");
    }
  }
}

int16_t feature_index(NnueFeatureBitmapType feature, unsigned square) {
  return feature * 64 + square;
}

int16_t flip_feature_index(int16_t index) {
  // Flip the board position vertically (rank 8 <-> rank 1, etc.) and swap colors.
  int16_t piece_type = (((index / 64) + (NF_COUNT / 2)) % NF_COUNT) * 64;
  int16_t square = index % 64;
  int16_t flipped_square = (7 - (square / 8)) * 8 + (square % 8);
  return piece_type + flipped_square;
}

Features pos2features(const struct ChessEngine::Position& pos) {
  ChessEngine::Threats threats;
  ChessEngine::create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
  Features features;
  for (unsigned i = 0; i < 64; ++i) {
    ChessEngine::SafeSquare sq = ChessEngine::SafeSquare(i);
    ChessEngine::ColoredPiece cp = pos.tiles_[sq];
    if (cp == ChessEngine::ColoredPiece::NO_COLORED_PIECE) {
      continue;
    }
    features.addFeature(feature_index(cp2nfbt(cp), sq));
    if (threats.badForCp(cp) & bb(sq)) {
      if (cp2color(cp) == ChessEngine::Color::WHITE) {
        features.addFeature(feature_index(NF_WHITE_HANGING_PIECES, sq));
      } else {
        features.addFeature(feature_index(NF_BLACK_HANGING_PIECES, sq));
      }
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