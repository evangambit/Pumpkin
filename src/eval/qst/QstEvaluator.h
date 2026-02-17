#ifndef SRC_EVAL_QST_QSTEVALUATOR_H
#define SRC_EVAL_QST_QSTEVALUATOR_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <bit>

#include "../../game/Position.h"
#include "../Evaluator.h"
#include "../OrientedBitboard.h"
#include "../../game/Utils.h"
#include "../../game/Threats.h"
#include "../PawnAnalysis.h"

namespace ChessEngine {

template<size_t N>
float min(const float *arr) {
  float m = arr[0];
  for (size_t i = 1; i < N; ++i) {
    if (arr[i] < m) {
      m = arr[i];
    }
  }
  return m;
}

template<size_t N>
float max(const float *arr) {
  float m = arr[0];
  for (size_t i = 1; i < N; ++i) {
    if (arr[i] > m) {
      m = arr[i];
    }
  }
  return m;
}

template<size_t N>
inline void load_flat(std::istream& in, float *out, const std::string& name) {
  char nameBuf[16];
  int32_t shapeLen;
  int32_t shape[1];

  in.read(nameBuf, 16);
  if (std::string(nameBuf, 16).find(name) == std::string::npos) {
    throw std::runtime_error("Expected name " + name + ", got " + std::string(nameBuf, 16));
  }
  in.read(reinterpret_cast<char*>(&shapeLen), sizeof(int32_t));
  if (shapeLen != 1) {
    throw std::runtime_error("Unexpected shape length " + std::to_string(shapeLen));
  }
  in.read(reinterpret_cast<char*>(shape), sizeof(int32_t) * shapeLen);
  if (shape[0] != N) {
    throw std::runtime_error("Unexpected shape " + std::to_string(shape[0]));
  }
  in.read(reinterpret_cast<char*>(out), sizeof(float) * N);
  for (size_t i = 0; i < N; ++i) {
    out[i] *= 200.0f;
  }
}


template<size_t QUANTIZATION>
struct QuantizedSquareTable {
  static_assert(QUANTIZATION > 0, "QUANTIZATION must be greater than 0");
  Evaluation weights[QUANTIZATION];
  Bitboard masks[QUANTIZATION];
  QuantizedSquareTable() {
    for (size_t i = 0; i < QUANTIZATION; ++i) {
        weights[i] = 0;
        masks[i] = 0;
    }
  }

  template<int SCALE = 1>
  inline void contribute(OrientedBitboard occupied, Evaluation *eval) const {
    static_assert(SCALE == 1 || SCALE == -1, "SCALE must be 1 or -1");
    Evaluation delta = 0;
    for (size_t i = 0; i < QUANTIZATION; ++i) {
      delta += std::popcount(occupied.bits & masks[i]) * weights[i];
    }
    if constexpr (SCALE == 1) {
      *eval += delta;
    } else {
      *eval -= delta;
    }
  }

  friend std::ostream& operator<<(std::ostream& stream, const QuantizedSquareTable<QUANTIZATION>& qst) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        size_t idx = y * 8 + x;
        float value = 0.0f;
        for (size_t i = 0; i < QUANTIZATION; ++i) {
          if ((qst.masks[i] >> idx) & 1) {
            value += static_cast<float>(qst.weights[i]);
          }
        }
        stream << rjust(std::to_string(int32_t(value)), 5) << " ";
      }
      stream << std::endl;
    }
    return stream;
  }
};

/**
 * Represents a quantized square table.
 */
template<size_t QUANTIZATION>
QuantizedSquareTable<QUANTIZATION> quantize(const float *weights) {
  if (QUANTIZATION == 64) {
    QuantizedSquareTable<QUANTIZATION> qst;
    for (size_t i = 0; i < 64; ++i) {
      qst.masks[i] = (1ULL << i);
      qst.weights[i] = static_cast<Evaluation>(std::round(weights[i]));
    }
    return qst;
  }

  // We use k-means clustering to quantize the weights.

  // A "0" centroid is free, due to how we use these values later, so we add
  // one extra centroid for zero.
  float means[QUANTIZATION + 1];
  constexpr size_t ZERO_INDEX = QUANTIZATION;
  means[ZERO_INDEX] = 0.0f;

  // Initialize centroids uniformly over the range of weights.
  float minWeight = min<64>(weights);
  float maxWeight = max<64>(weights);
  float range = maxWeight - minWeight + 1e-5f;
  for (size_t i = 0; i < QUANTIZATION; ++i) {
    means[i] = minWeight + range * (i + 0.5f) / QUANTIZATION;
  }

  // Run k-means for a fixed number of iterations.
  const size_t kNumIterations = 10;
  for (size_t iter = 0; iter < kNumIterations; ++iter) {
    float newMeans[QUANTIZATION + 1] = {0};
    size_t counts[QUANTIZATION + 1] = {0};
    for (size_t i = 0; i < 64; ++i) {
      // Find closest centroid.
      size_t bestCentroid = ZERO_INDEX;
      float bestDist = std::abs(weights[i] - means[ZERO_INDEX]);
      for (size_t c = 0; c < QUANTIZATION; ++c) {
        float dist = std::abs(weights[i] - means[c]);
        if (dist < bestDist) {
          bestDist = dist;
          bestCentroid = c;
        }
      }
      newMeans[bestCentroid] += weights[i];
      counts[bestCentroid] += 1;
    }
    // Update centroids.
    for (size_t c = 0; c < QUANTIZATION; ++c) {
      if (counts[c] > 0) {
        means[c] = newMeans[c] / counts[c];
      }
    }
    means[ZERO_INDEX] = 0.0f;  // Unneeded, but for clarity.
  }

  // Build QuantizedSquareTable from centroids.
  QuantizedSquareTable<QUANTIZATION> qst;
  for (size_t i = 0; i < 64; ++i) {
    // Find closest centroid.
    size_t bestCentroid = ZERO_INDEX;
    float bestDist = std::abs(weights[i] - means[ZERO_INDEX]);
    for (size_t c = 0; c < QUANTIZATION; ++c) {
      float dist = std::abs(weights[i] - means[c]);
      if (dist < bestDist) {
        bestDist = dist;
        bestCentroid = c;
      }
    }
    if (bestCentroid != ZERO_INDEX) {
      qst.masks[bestCentroid] |= (1ULL << i);
    }
  }

  for (size_t c = 0; c < QUANTIZATION; ++c) {
    qst.weights[c] = static_cast<Evaluation>(std::round(means[c]));
  }
  return qst;
}

template<size_t QUANTIZATION>
QuantizedSquareTable<QUANTIZATION> load_quantized_square_table(std::istream& in, const std::string& name) {
  float out[64];
  char nameBuf[16];
  int32_t shapeLen;
  int32_t shape[2];

  in.read(nameBuf, 16);
  if (std::string(nameBuf, 16).find(name) == std::string::npos) {
    throw std::runtime_error("Expected name " + name + ", got " + std::string(nameBuf, 16));
  }
  in.read(reinterpret_cast<char*>(&shapeLen), sizeof(int32_t));
  if (shapeLen != 2) {
    throw std::runtime_error("Unexpected shape length " + std::to_string(shapeLen));
  }
  in.read(reinterpret_cast<char*>(shape), sizeof(int32_t) * shapeLen);
  if (shape[0] != 8 || shape[1] != 8) {
    throw std::runtime_error("Unexpected shape " + std::to_string(shape[0]) + "," + std::to_string(shape[1]));
  }
  in.read(reinterpret_cast<char*>(out), sizeof(float) * 64);
  for (size_t i = 0; i < 64; ++i) {
    out[i] *= 200.0f;
  }
  return quantize<QUANTIZATION>(out);
}

/**
  * Two quantized square tables, one for early game and one for late game.
  */
template<size_t QUANTIZATION>
struct TaperedQuantizedSquareTable {
  QuantizedSquareTable<QUANTIZATION> earlyWeights;
  QuantizedSquareTable<QUANTIZATION> lateWeights;

  template<int SCALE = 1>
  void contribute(OrientedBitboard occupied, Evaluation *early, Evaluation *late) const {
    if (QUANTIZATION == 0) {
      return;
    }
    earlyWeights.template contribute<SCALE>(occupied, early);
    lateWeights.template contribute<SCALE>(occupied, late);
  }

  void load_table(std::istream& in, const std::string& name) {
    earlyWeights = load_quantized_square_table<QUANTIZATION>(in, "e_" + name);
    lateWeights = load_quantized_square_table<QUANTIZATION>(in, "l_" + name);
  }
};

/**
 * Shift a bitboard s.t. the source square maps to the destination square.
 */
template<SafeSquare DESTINATION>
inline Bitboard shiftToDestination(SafeSquare source, Bitboard bb) {
  const int dx = DESTINATION % 8 - source % 8;
  const int dy = DESTINATION / 8 - source / 8;
  if (dx > 0) {
    bb = shift_ew<Direction::EAST>(bb, dx);
  } else if (dx < 0) {
    bb = shift_ew<Direction::WEST>(bb, -dx);
  }
  if (dy > 0) {
    bb <<= dy * 8;
  } else if (dy < 0) {
    bb >>= -dy * 8;
  }
  return bb;
}
template<SafeSquare DESTINATION>
inline OrientedBitboard shiftToDestination(SafeSquare source, OrientedBitboard ob) {
  return OrientedBitboard{shiftToDestination<DESTINATION>(source, ob.bits)};
}


enum QstFeatures {
  Q_PAWNS_US = 0,
  Q_PAWNS_THEM = 1,
  Q_KNIGHTS_US,
  Q_KNIGHTS_THEM,
  Q_BISHOPS_US,
  Q_BISHOPS_THEM,
  Q_ROOKS_US,
  Q_ROOKS_THEM,
  Q_QUEENS_US,
  Q_QUEENS_THEM,
  Q_KINGS_US,
  Q_KINGS_THEM,
  Q_KING_NO_CASTLE_US,  // 12
  Q_KING_NO_CASTLE_THEM,
  Q_PASSED_PAWNS_US,
  Q_PASSED_PAWNS_THEM,
  Q_ISOLATED_PAWNS_US,
  Q_ISOLATED_PAWNS_THEM,
  Q_DOUBLED_PAWNS_US,
  Q_DOUBLED_PAWNS_THEM,
  Q_BAD_FOR_PAWN_US,
  Q_BAD_FOR_PAWN_THEM,
  Q_BAD_FOR_KNIGHT_US,
  Q_BAD_FOR_KNIGHT_THEM,
  Q_BAD_FOR_BISHOP_US,
  Q_BAD_FOR_BISHOP_THEM,
  Q_BAD_FOR_ROOK_US,
  Q_BAD_FOR_ROOK_THEM,
  Q_BAD_FOR_QUEEN_US,
  Q_BAD_FOR_QUEEN_THEM,
  Q_BAD_FOR_KING_US,
  Q_BAD_FOR_KING_THEM,
  Q_HANGING_PAWN_US,
  Q_HANGING_PAWN_THEM,
  Q_HANGING_KNIGHT_US,
  Q_HANGING_KNIGHT_THEM,
  Q_HANGING_BISHOP_US,
  Q_HANGING_BISHOP_THEM,
  Q_HANGING_ROOK_US,
  Q_HANGING_ROOK_THEM,
  Q_HANGING_QUEEN_US,
  Q_HANGING_QUEEN_THEM,
  Q_HANGING_KING_US,
  Q_HANGING_KING_THEM,
  Q_BAD_FOR_PAWN_NEAR_KING_US,
  Q_BAD_FOR_PAWN_NEAR_KING_THEM,
  Q_BAD_FOR_BISHOP_NEAR_KING_US,
  Q_BAD_FOR_BISHOP_NEAR_KING_THEM,
  Q_BAD_FOR_KING_NEAR_KING_US,
  Q_BAD_FOR_KING_NEAR_KING_THEM,
  Q_PAWN_IN_FRONT_OF_KING_US,
  Q_PAWN_IN_FRONT_OF_KING_THEM,
  Q_PAWN_STORM_US,
  Q_PAWN_STORM_THEM,
  Q_ADJACENT_PAWNS_US,
  Q_ADJACENT_PAWNS_THEM,
  Q_DIAGONAL_PAWNS_US,
  Q_DIAGONAL_PAWNS_THEM,
  Q_NUM_FEATURES
};

inline std::string feature_name(QstFeatures f) {
  switch (f) {
    case Q_PAWNS_US: return "Q_PAWNS_US";
    case Q_PAWNS_THEM: return "Q_PAWNS_THEM";
    case Q_KNIGHTS_US: return "Q_KNIGHTS_US";
    case Q_KNIGHTS_THEM: return "Q_KNIGHTS_THEM";
    case Q_BISHOPS_US: return "Q_BISHOPS_US";
    case Q_BISHOPS_THEM: return "Q_BISHOPS_THEM";
    case Q_ROOKS_US: return "Q_ROOKS_US";
    case Q_ROOKS_THEM: return "Q_ROOKS_THEM";
    case Q_QUEENS_US: return "Q_QUEENS_US";
    case Q_QUEENS_THEM: return "Q_QUEENS_THEM";
    case Q_KINGS_US: return "Q_KINGS_US";
    case Q_KINGS_THEM: return "Q_KINGS_THEM";
    case Q_KING_NO_CASTLE_US: return "Q_KING_NO_CASTLE_US";
    case Q_KING_NO_CASTLE_THEM: return "Q_KING_NO_CASTLE_THEM";
    case Q_PASSED_PAWNS_US: return "Q_PASSED_PAWNS_US";
    case Q_PASSED_PAWNS_THEM: return "Q_PASSED_PAWNS_THEM";
    case Q_ISOLATED_PAWNS_US: return "Q_ISOLATED_PAWNS_US";
    case Q_ISOLATED_PAWNS_THEM: return "Q_ISOLATED_PAWNS_THEM";
    case Q_DOUBLED_PAWNS_US: return "Q_DOUBLED_PAWNS_US";
    case Q_DOUBLED_PAWNS_THEM: return "Q_DOUBLED_PAWNS_THEM";
    case Q_BAD_FOR_PAWN_US: return "Q_BAD_FOR_PAWN_US";
    case Q_BAD_FOR_PAWN_THEM: return "Q_BAD_FOR_PAWN_THEM";
    case Q_BAD_FOR_KNIGHT_US: return "Q_BAD_FOR_KNIGHT_US";
    case Q_BAD_FOR_KNIGHT_THEM: return "Q_BAD_FOR_KNIGHT_THEM";
    case Q_BAD_FOR_BISHOP_US: return "Q_BAD_FOR_BISHOP_US";
    case Q_BAD_FOR_BISHOP_THEM: return "Q_BAD_FOR_BISHOP_THEM";
    case Q_BAD_FOR_ROOK_US: return "Q_BAD_FOR_ROOK_US";
    case Q_BAD_FOR_ROOK_THEM: return "Q_BAD_FOR_ROOK_THEM";
    case Q_BAD_FOR_QUEEN_US: return "Q_BAD_FOR_QUEEN_US";
    case Q_BAD_FOR_QUEEN_THEM: return "Q_BAD_FOR_QUEEN_THEM";
    case Q_BAD_FOR_KING_US: return "Q_BAD_FOR_KING_US";
    case Q_BAD_FOR_KING_THEM: return "Q_BAD_FOR_KING_THEM";
    case Q_HANGING_PAWN_US: return "Q_HANGING_PAWN_US";
    case Q_HANGING_PAWN_THEM: return "Q_HANGING_PAWN_THEM";
    case Q_HANGING_KNIGHT_US: return "Q_HANGING_KNIGHT_US";
    case Q_HANGING_KNIGHT_THEM: return "Q_HANGING_KNIGHT_THEM";
    case Q_HANGING_BISHOP_US: return "Q_HANGING_BISHOP_US";
    case Q_HANGING_BISHOP_THEM: return "Q_HANGING_BISHOP_THEM";
    case Q_HANGING_ROOK_US: return "Q_HANGING_ROOK_US";
    case Q_HANGING_ROOK_THEM: return "Q_HANGING_ROOK_THEM";
    case Q_HANGING_QUEEN_US: return "Q_HANGING_QUEEN_US";
    case Q_HANGING_QUEEN_THEM: return "Q_HANGING_QUEEN_THEM";
    case Q_HANGING_KING_US: return "Q_HANGING_KING_US";
    case Q_HANGING_KING_THEM: return "Q_HANGING_KING_THEM";
    case Q_BAD_FOR_PAWN_NEAR_KING_US: return "Q_BAD_FOR_PAWN_NEAR_KING_US";
    case Q_BAD_FOR_PAWN_NEAR_KING_THEM: return "Q_BAD_FOR_PAWN_NEAR_KING_THEM";
    case Q_BAD_FOR_BISHOP_NEAR_KING_US: return "Q_BAD_FOR_BISHOP_NEAR_KING_US";
    case Q_BAD_FOR_BISHOP_NEAR_KING_THEM: return "Q_BAD_FOR_BISHOP_NEAR_KING_THEM";
    case Q_BAD_FOR_KING_NEAR_KING_US: return "Q_BAD_FOR_KING_NEAR_KING_US";
    case Q_BAD_FOR_KING_NEAR_KING_THEM: return "Q_BAD_FOR_KING_NEAR_KING_THEM";
    case Q_PAWN_IN_FRONT_OF_KING_US: return "Q_PAWN_IN_FRONT_OF_KING_US";
    case Q_PAWN_IN_FRONT_OF_KING_THEM: return "Q_PAWN_IN_FRONT_OF_KING_THEM";
    case Q_PAWN_STORM_US: return "Q_PAWN_STORM_US";
    case Q_PAWN_STORM_THEM: return "Q_PAWN_STORM_THEM";
    case Q_ADJACENT_PAWNS_US: return "Q_ADJACENT_PAWNS_US";
    case Q_ADJACENT_PAWNS_THEM: return "Q_ADJACENT_PAWNS_THEM";
    case Q_DIAGONAL_PAWNS_US: return "Q_DIAGONAL_PAWNS_US";
    case Q_DIAGONAL_PAWNS_THEM: return "Q_DIAGONAL_PAWNS_THEM";
    default: return "UNKNOWN_FEATURE";
  }
}

/**
 * Quantized Square Table Evaluator
 *
 * Uses an un-quantized piece-square table for evaluation, 
 * plus a host of quantized, conditional square tables for finer adjustments.
 *
 * Weights are flipped-and-negatived for the opponent, so we only need to store
 * tables from one side's perspective.
 *
 * In contrast, features are bitmaps, so we need to produce two sets of features
 * for each quantized square table: one for white and one for black. On the other
 * hand, many of our tables are "tapered", meaning they have early-game and late-game
 * weights.
 */
struct QstEvaluator : public EvaluatorInterface {
  QstEvaluator() {}

  void load(std::string filename) {
    std::ifstream in(filename, std::ios::binary);
    this->load(in);
    in.close();
  }

  void load(std::istream& in) {
    for (size_t i = 0; i < 6; ++i) {
      pieces[i].load_table(in, std::string(1, "pnbrqk"[i]) + "|base_psq");
    }
    kingAssumingNoCastling.load_table(in, "k|no_castle");
    passedPawns.load_table(in, "passed_pawns");
    isolatedPawns.load_table(in, "isolated_pawns");
    doubledPawns.load_table(in, "doubled_pawns");

    for (size_t i = 0; i < 6; ++i) {
      control[i].load_table(in, "bad_for_" + std::string(1, "pnbrqk"[i]));
    }
    for (size_t i = 0; i < 6; ++i) {
      capturable[i].load_table(in, "hanging_" + std::string(1, "pnbrqk"[i]));
    }
    
    badSqNearKing[0].load_table(in, "badSqNearK_p");
    badSqNearKing[1].load_table(in, "badSqNearK_b");
    badSqNearKing[2].load_table(in, "badSqNearK_k");
    pInFrontOfK.load_table(in, "pInFrontOfK");
    pawnStorm.load_table(in, "pawnStorm");

    adjacentPawns.load_table(in, "adjacentPawns");
    diagonalPawns.load_table(in, "diagonalPawns");

    float biasWeights[2];
    load_flat<2>(in, biasWeights, "biases");
    biases[0] = static_cast<Evaluation>(std::round(biasWeights[0]));
    biases[1] = static_cast<Evaluation>(std::round(biasWeights[1]));
  }

  // Normal piece-square tables.
  TaperedQuantizedSquareTable<8> pieces[6];
  
  TaperedQuantizedSquareTable<8> kingAssumingNoCastling;

  TaperedQuantizedSquareTable<2> passedPawns;
  TaperedQuantizedSquareTable<2> isolatedPawns;
  TaperedQuantizedSquareTable<2> doubledPawns;

  // Whether a square is "bad" for a type of piece to be sitting.
  // Maybe more intuitively: a square where a pawn cannot safely exist is strongly
  // under the influence of enemy pieces. A square where a rook cannot safely exist
  // is moderately under the influence of enemy pieces. etc.
  TaperedQuantizedSquareTable<2> control[6];

  // Bitboard for each piece type, indicating whether they're in danger of being captured.
  TaperedQuantizedSquareTable<2> capturable[6];

  // King safety features.
  TaperedQuantizedSquareTable<2> badSqNearKing[3];
  TaperedQuantizedSquareTable<2> pInFrontOfK;
  TaperedQuantizedSquareTable<2> pawnStorm;

  TaperedQuantizedSquareTable<2> adjacentPawns;
  TaperedQuantizedSquareTable<2> diagonalPawns;

  Evaluation biases[2];

  // Extract features from the position from mover's perspective.
  // This means the mover's pawns are always moving NORTH. This is
  // accomplished by flipping all bitboards vertically if the mover is black
  // (see the end of this function).
  template<Color US>
  void get_features(const Position& pos, std::vector<Bitboard> *out) {
    this->evaluate<US>(pos);
    for (size_t i = 0; i < Q_NUM_FEATURES; ++i) {
      out->push_back(features[i].bits);
    }
  }

  // It's a little hacky to store features as a member variable, but it helps
  // ensure that get_features and evaluate are consistent with each other.
  // Unfortunately this means
  OrientedBitboard features[Q_NUM_FEATURES];

  template<Color US>
  ColoredEvaluation<US> evaluate(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
    constexpr Direction kBackward = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;

    const OrientedBitboard ourPawns = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    const OrientedBitboard theirPawns = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);

    int32_t stage = earliness(pos);
    Evaluation early = biases[0];
    Evaluation late = biases[1];

    PawnAnalysis<US> pawnAnalysis(pos);

    features[Q_PAWNS_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    features[Q_PAWNS_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    pieces[0].contribute(features[Q_PAWNS_US], &early, &late);
    pieces[0].contribute<-1>(flip_vertically(features[Q_PAWNS_THEM]), &early, &late);

    features[Q_KNIGHTS_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]);
    features[Q_KNIGHTS_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]);
    pieces[1].contribute(features[Q_KNIGHTS_US], &early, &late);
    pieces[1].contribute<-1>(flip_vertically(features[Q_KNIGHTS_THEM]), &early, &late);

    features[Q_BISHOPS_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    features[Q_BISHOPS_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    pieces[2].contribute(features[Q_BISHOPS_US], &early, &late);
    pieces[2].contribute<-1>(flip_vertically(features[Q_BISHOPS_THEM]), &early, &late);

    features[Q_ROOKS_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    features[Q_ROOKS_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    pieces[3].contribute(features[Q_ROOKS_US], &early, &late);
    pieces[3].contribute<-1>(flip_vertically(features[Q_ROOKS_THEM]), &early, &late);

    features[Q_QUEENS_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]);
    features[Q_QUEENS_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]);
    pieces[4].contribute(features[Q_QUEENS_US], &early, &late);
    pieces[4].contribute<-1>(flip_vertically(features[Q_QUEENS_THEM]), &early, &late);

    features[Q_KINGS_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    features[Q_KINGS_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);
    pieces[5].contribute(features[Q_KINGS_US], &early, &late);
    pieces[5].contribute<-1>(flip_vertically(features[Q_KINGS_THEM]), &early, &late);

    // kingAssumingNoCastling contributions
    constexpr uint8_t ourKingsideCastling = US == Color::WHITE ? kCastlingRights_WhiteKing : kCastlingRights_BlackKing;
    constexpr uint8_t ourQueensideCastling = US == Color::WHITE ? kCastlingRights_WhiteQueen : kCastlingRights_BlackQueen;
    constexpr uint8_t theirKingsideCastling = US == Color::WHITE ? kCastlingRights_BlackKing : kCastlingRights_WhiteKing;
    constexpr uint8_t theirQueensideCastling = US == Color::WHITE ? kCastlingRights_BlackQueen : kCastlingRights_WhiteQueen;
    const bool weCannotCastle = (pos.currentState_.castlingRights & (ourKingsideCastling | ourQueensideCastling)) == 0;
    const bool theyCannotCastle = (pos.currentState_.castlingRights & (theirKingsideCastling | theirQueensideCastling)) == 0;
    features[Q_KING_NO_CASTLE_US] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]).iff(weCannotCastle);
    features[Q_KING_NO_CASTLE_THEM] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]).iff(theyCannotCastle);
    kingAssumingNoCastling.contribute(features[Q_KING_NO_CASTLE_US], &early, &late);
    kingAssumingNoCastling.contribute<-1>(flip_vertically(features[Q_KING_NO_CASTLE_THEM]), &early, &late);

    features[Q_PASSED_PAWNS_US] = orient<US>(pawnAnalysis.ourPassedPawns);
    features[Q_PASSED_PAWNS_THEM] = orient<US>(pawnAnalysis.theirPassedPawns);
    passedPawns.contribute(features[Q_PASSED_PAWNS_US], &early, &late);
    passedPawns.contribute<-1>(flip_vertically(features[Q_PASSED_PAWNS_THEM]), &early, &late);

    features[Q_ISOLATED_PAWNS_US] = orient<US>(pawnAnalysis.ourIsolatedPawns);
    features[Q_ISOLATED_PAWNS_THEM] = orient<US>(pawnAnalysis.theirIsolatedPawns);
    isolatedPawns.contribute(features[Q_ISOLATED_PAWNS_US], &early, &late);
    isolatedPawns.contribute<-1>(flip_vertically(features[Q_ISOLATED_PAWNS_THEM]), &early, &late);


    features[Q_DOUBLED_PAWNS_US] = orient<US>(pawnAnalysis.ourDoubledPawns);
    features[Q_DOUBLED_PAWNS_THEM] = orient<US>(pawnAnalysis.theirDoubledPawns);
    doubledPawns.contribute(features[Q_DOUBLED_PAWNS_US], &early, &late);
    doubledPawns.contribute<-1>(flip_vertically(features[Q_DOUBLED_PAWNS_THEM]), &early, &late);

    Threats threats(pos.pieceBitboards_, pos.colorBitboards_);
    features[Q_BAD_FOR_PAWN_US] = orient<US>(threats.badForOur<US>(Piece::PAWN));
    features[Q_BAD_FOR_PAWN_THEM] = orient<US>(threats.badForOur<THEM>(Piece::PAWN));
    features[Q_BAD_FOR_KNIGHT_US] = orient<US>(threats.badForOur<US>(Piece::KNIGHT));
    features[Q_BAD_FOR_KNIGHT_THEM] = orient<US>(threats.badForOur<THEM>(Piece::KNIGHT));
    features[Q_BAD_FOR_BISHOP_US] = orient<US>(threats.badForOur<US>(Piece::BISHOP));
    features[Q_BAD_FOR_BISHOP_THEM] = orient<US>(threats.badForOur<THEM>(Piece::BISHOP));
    features[Q_BAD_FOR_ROOK_US] = orient<US>(threats.badForOur<US>(Piece::ROOK));
    features[Q_BAD_FOR_ROOK_THEM] = orient<US>(threats.badForOur<THEM>(Piece::ROOK));
    features[Q_BAD_FOR_QUEEN_US] = orient<US>(threats.badForOur<US>(Piece::QUEEN));
    features[Q_BAD_FOR_QUEEN_THEM] = orient<US>(threats.badForOur<THEM>(Piece::QUEEN));
    features[Q_BAD_FOR_KING_US] = orient<US>(threats.badForOur<US>(Piece::KING));
    features[Q_BAD_FOR_KING_THEM] = orient<US>(threats.badForOur<THEM>(Piece::KING));
    control[0].contribute(features[Q_BAD_FOR_PAWN_US], &early, &late);
    control[0].contribute<-1>(flip_vertically(features[Q_BAD_FOR_PAWN_THEM]), &early, &late);
    control[1].contribute(features[Q_BAD_FOR_KNIGHT_US], &early, &late);
    control[1].contribute<-1>(flip_vertically(features[Q_BAD_FOR_KNIGHT_THEM]), &early, &late);
    control[2].contribute(features[Q_BAD_FOR_BISHOP_US], &early, &late);
    control[2].contribute<-1>(flip_vertically(features[Q_BAD_FOR_BISHOP_THEM]), &early, &late);
    control[3].contribute(features[Q_BAD_FOR_ROOK_US], &early, &late);
    control[3].contribute<-1>(flip_vertically(features[Q_BAD_FOR_ROOK_THEM]), &early, &late);
    control[4].contribute(features[Q_BAD_FOR_QUEEN_US], &early, &late);
    control[4].contribute<-1>(flip_vertically(features[Q_BAD_FOR_QUEEN_THEM]), &early, &late);
    control[5].contribute(features[Q_BAD_FOR_KING_US], &early, &late);
    control[5].contribute<-1>(flip_vertically(features[Q_BAD_FOR_KING_THEM]), &early, &late);

    // capturable contributions (pieces that are in danger of being captured)
    features[Q_HANGING_PAWN_US] = orient<US>(threats.badForOur<US>(Piece::PAWN) & pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    features[Q_HANGING_PAWN_THEM] = orient<US>(threats.badForOur<THEM>(Piece::PAWN) & pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    features[Q_HANGING_KNIGHT_US] = orient<US>(threats.badForOur<US>(Piece::KNIGHT) & pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]);
    features[Q_HANGING_KNIGHT_THEM] = orient<US>(threats.badForOur<THEM>(Piece::KNIGHT) & pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]);
    features[Q_HANGING_BISHOP_US] = orient<US>(threats.badForOur<US>(Piece::BISHOP) & pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    features[Q_HANGING_BISHOP_THEM] = orient<US>(threats.badForOur<THEM>(Piece::BISHOP) & pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    features[Q_HANGING_ROOK_US] = orient<US>(threats.badForOur<US>(Piece::ROOK) & pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    features[Q_HANGING_ROOK_THEM] = orient<US>(threats.badForOur<THEM>(Piece::ROOK) & pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    features[Q_HANGING_QUEEN_US] = orient<US>(threats.badForOur<US>(Piece::QUEEN) & pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]);
    features[Q_HANGING_QUEEN_THEM] = orient<US>(threats.badForOur<THEM>(Piece::QUEEN) & pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]);
    features[Q_HANGING_KING_US] = orient<US>(threats.badForOur<US>(Piece::KING) & pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    features[Q_HANGING_KING_THEM] = orient<US>(threats.badForOur<THEM>(Piece::KING) & pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);
    capturable[0].contribute(features[Q_HANGING_PAWN_US], &early, &late);
    capturable[0].contribute<-1>(flip_vertically(features[Q_HANGING_PAWN_THEM]), &early, &late);
    capturable[1].contribute(features[Q_HANGING_KNIGHT_US], &early, &late);
    capturable[1].contribute<-1>(flip_vertically(features[Q_HANGING_KNIGHT_THEM]), &early, &late);
    capturable[2].contribute(features[Q_HANGING_BISHOP_US], &early, &late);
    capturable[2].contribute<-1>(flip_vertically(features[Q_HANGING_BISHOP_THEM]), &early, &late);
    capturable[3].contribute(features[Q_HANGING_ROOK_US], &early, &late);
    capturable[3].contribute<-1>(flip_vertically(features[Q_HANGING_ROOK_THEM]), &early, &late);
    capturable[4].contribute(features[Q_HANGING_QUEEN_US], &early, &late);
    capturable[4].contribute<-1>(flip_vertically(features[Q_HANGING_QUEEN_THEM]), &early, &late);
    capturable[5].contribute(features[Q_HANGING_KING_US], &early, &late);
    capturable[5].contribute<-1>(flip_vertically(features[Q_HANGING_KING_THEM]), &early, &late);

    SafeSquare ourKingSq = lsb_i_promise_board_is_not_empty(orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]));
    SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]));

    //
    // King safety. For king-specific features we shift all bitboards s.t. our king is on e4, so that the features are king-relative.
    //

    // How well do we control the squares around our king?

    const OrientedBitboard nearOurKingMask = OrientedBitboard{kKingDist[2][SafeSquare::SE4]};
    const OrientedBitboard nearTheirKingMask = OrientedBitboard{kKingDist[2][SafeSquare::SE5]};

    features[Q_BAD_FOR_PAWN_NEAR_KING_US] = shiftToDestination<SafeSquare::SE4>( ourKingSq, orient<US>(threats.badForOur<US>(Piece::PAWN))) & nearOurKingMask;
    features[Q_BAD_FOR_PAWN_NEAR_KING_THEM] = shiftToDestination<SafeSquare::SE5>(theirKingSq, orient<US>(threats.badForOur<THEM>(Piece::PAWN))) & nearTheirKingMask;
    features[Q_BAD_FOR_BISHOP_NEAR_KING_US] = shiftToDestination<SafeSquare::SE4>( ourKingSq, orient<US>(threats.badForOur<US>(Piece::BISHOP))) & nearOurKingMask;
    features[Q_BAD_FOR_BISHOP_NEAR_KING_THEM] = shiftToDestination<SafeSquare::SE5>(theirKingSq, orient<US>(threats.badForOur<THEM>(Piece::BISHOP))) & nearTheirKingMask;
    features[Q_BAD_FOR_KING_NEAR_KING_US] = shiftToDestination<SafeSquare::SE4>( ourKingSq, orient<US>(threats.badForOur<US>(Piece::KING))) & nearOurKingMask;
    features[Q_BAD_FOR_KING_NEAR_KING_THEM] = shiftToDestination<SafeSquare::SE5>(theirKingSq, orient<US>(threats.badForOur<THEM>(Piece::KING))) & nearTheirKingMask;
    badSqNearKing[0].contribute(features[Q_BAD_FOR_PAWN_NEAR_KING_US], &early, &late);
    badSqNearKing[0].contribute<-1>(flip_vertically(features[Q_BAD_FOR_PAWN_NEAR_KING_THEM]), &early, &late);
    badSqNearKing[1].contribute(features[Q_BAD_FOR_BISHOP_NEAR_KING_US], &early, &late);
    badSqNearKing[1].contribute<-1>(flip_vertically(features[Q_BAD_FOR_BISHOP_NEAR_KING_THEM]), &early, &late);
    badSqNearKing[2].contribute(features[Q_BAD_FOR_KING_NEAR_KING_US], &early, &late);
    badSqNearKing[2].contribute<-1>(flip_vertically(features[Q_BAD_FOR_KING_NEAR_KING_THEM]), &early, &late);

    // Pawns in front of the king.
    features[Q_PAWN_IN_FRONT_OF_KING_US] = shiftToDestination<SafeSquare::SE4>( ourKingSq, orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()])) & nearOurKingMask;
    features[Q_PAWN_IN_FRONT_OF_KING_THEM] = shiftToDestination<SafeSquare::SE5>(theirKingSq, orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()])) & nearTheirKingMask;
    pInFrontOfK.contribute(features[Q_PAWN_IN_FRONT_OF_KING_US], &early, &late);
    pInFrontOfK.contribute<-1>(flip_vertically(features[Q_PAWN_IN_FRONT_OF_KING_THEM]), &early, &late);

    // Pawn storms.
    features[Q_PAWN_STORM_US] = shiftToDestination<SafeSquare::SE4>( ourKingSq, theirPawns);
    features[Q_PAWN_STORM_THEM] = shiftToDestination<SafeSquare::SE5>(theirKingSq, ourPawns);
    pawnStorm.contribute(features[Q_PAWN_STORM_US], &early, &late);
    pawnStorm.contribute<-1>(flip_vertically(features[Q_PAWN_STORM_THEM]), &early, &late);

    // adjacentPawns
    features[Q_ADJACENT_PAWNS_US] = shift<Direction::EAST>(ourPawns) & ourPawns;
    features[Q_ADJACENT_PAWNS_THEM] = shift<Direction::EAST>(theirPawns) & theirPawns;
    adjacentPawns.contribute(features[Q_ADJACENT_PAWNS_US], &early, &late);
    adjacentPawns.contribute<-1>(flip_vertically(features[Q_ADJACENT_PAWNS_THEM]), &early, &late);

    // diagonalPawns
    features[Q_DIAGONAL_PAWNS_US] = shift<Direction::NORTH_EAST>(ourPawns) & ourPawns;
    features[Q_DIAGONAL_PAWNS_THEM] = shift<Direction::SOUTH_EAST>(theirPawns) & theirPawns;
    diagonalPawns.contribute(features[Q_DIAGONAL_PAWNS_US], &early, &late);
    diagonalPawns.contribute<-1>(flip_vertically(features[Q_DIAGONAL_PAWNS_THEM]), &early, &late);

    int32_t eval = int32_t(early) * stage + int32_t(late) * (18 - stage);
    return ColoredEvaluation<US>(eval / 18);
  }

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    return evaluate<Color::WHITE>(pos);
  }

  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    return evaluate<Color::BLACK>(pos);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<QstEvaluator>();
  }

  void empty() override {
  }
  void place_piece(ColoredPiece cp, SafeSquare square) override {
  }
  void remove_piece(ColoredPiece cp, SafeSquare square) override {
  }

  void place_piece(SafeColoredPiece cp, SafeSquare square) override {
  }
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {
  }

  std::string to_string() const override {
    return "QstEvaluator";
  }

 private:
  inline int32_t earliness(const Position& pos) const {
    int32_t t = 0;
    t += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]) * 4;
    t += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]) * 4;
    t += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]) * 1;
    t += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]) * 1;
    t += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]) * 1;
    t += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]) * 1;
    return t;
  }
};

} // namespace ChessEngine

#endif  // SRC_EVAL_QST_QSTEVALUATOR_H
