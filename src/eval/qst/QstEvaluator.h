#ifndef SRC_EVAL_QST_QSTEVALUATOR_H
#define SRC_EVAL_QST_QSTEVALUATOR_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <bit>

#include "../../game/Position.h"
#include "../evaluator.h"
#include "../../game/utils.h"
#include "../../game/Threats.h"

namespace ChessEngine {

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
}

inline void load64(std::istream& in, float *out, const std::string& name) {
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
}

template<Color US>
struct PawnAnalysis {
  Bitboard ourPassedPawns, theirPassedPawns;
  Bitboard ourIsolatedPawns, theirIsolatedPawns;
  Bitboard ourDoubledPawns, theirDoubledPawns;

  PawnAnalysis(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
    constexpr Direction kBackward = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;

    Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    
    Bitboard aheadOfOurPawns = US == Color::WHITE ? northFill(ourPawns) : southFill(ourPawns);
    Bitboard aheadOfTheirPawns = US == Color::WHITE ? southFill(theirPawns) : northFill(theirPawns);
    Bitboard filesWithOurPawns = US == Color::WHITE ? southFill(aheadOfOurPawns) : northFill(aheadOfOurPawns);
    Bitboard filesWithTheirPawns = US == Color::WHITE ? northFill(aheadOfTheirPawns) : southFill(aheadOfTheirPawns);
    Bitboard filesWithoutOurPawns = ~filesWithOurPawns;
    Bitboard filesWithoutTheirPawns = ~filesWithTheirPawns;
    this->ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(aheadOfTheirPawns));
    this->theirPassedPawns = theirPawns & ~shift<kForward>(fatten(aheadOfOurPawns));
    this->ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
    this->theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
    this->ourDoubledPawns = ourPawns & shift<kForward>(aheadOfOurPawns);
    this->theirDoubledPawns = theirPawns & shift<kBackward>(aheadOfTheirPawns);
  }
};

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

  template<Color US>
  inline void contribute(Bitboard occupied, Evaluation *eval) const {
    if constexpr (US == Color::BLACK) {
      occupied = flip_vertically(occupied);
    }
    for (size_t i = 0; i < QUANTIZATION; ++i) {
      *eval -= std::popcount(occupied & masks[i]) * weights[i];
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

/**
 * Quantize a 64-element array of weights into a QuantizedSquareTable
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
struct TaperedQuantizedSquareTable {
  QuantizedSquareTable<QUANTIZATION> earlyWeights;
  QuantizedSquareTable<QUANTIZATION> lateWeights;

  void load(std::istream& in) {
    float weights[128];
    in.read(reinterpret_cast<char*>(weights), sizeof(float) * 128);
    // Scale weights for better quantization.
    for (size_t i = 0; i < 128; ++i) {
      weights[i] *= 200.0f;
    }
    earlyWeights = quantize<QUANTIZATION>(weights);
    lateWeights = quantize<QUANTIZATION>(weights + 64);
  }

  template<Color US>
  void contribute(Bitboard occupied, Evaluation *early, Evaluation *late) const {
    if (QUANTIZATION == 0) {
      return;
    }
    earlyWeights.template contribute<US>(occupied, early);
    lateWeights.template contribute<US>(occupied, late);
  }

  static TaperedQuantizedSquareTable<QUANTIZATION> load_table(std::istream& in, const std::string& name) {
    float weights[128];

    TaperedQuantizedSquareTable<QUANTIZATION> out;

    load64(in, weights, "e_" + name);
    load64(in, weights + 64, "l_" + name);

    // Scale weights for better quantization.
    for (size_t i = 0; i < 128; ++i) {
      weights[i] *= 200.0f;
    }
    out.earlyWeights = quantize<QUANTIZATION>(weights);
    out.lateWeights = quantize<QUANTIZATION>(weights + 64);
    return out;
  }
};

template<size_t QUANTIZATION>
struct ConditionedTaperedPieceSquareTables {
  TaperedQuantizedSquareTable<QUANTIZATION> tables[12];
  void load_table(std::istream& in, const std::string& base_name) {
    tables[0] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_p|" + base_name);
    tables[1] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_n|" + base_name);
    tables[2] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_b|" + base_name);
    tables[3] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_r|" + base_name);
    tables[4] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_q|" + base_name);
    tables[5] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_k|" + base_name);
    tables[6] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_p|" + base_name);
    tables[7] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_n|" + base_name);
    tables[8] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_b|" + base_name);
    tables[9] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_r|" + base_name);
    tables[10] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_q|" + base_name);
    tables[11] = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_k|" + base_name);
  }

  template<Color US>
  void contribute(const Position& pos, bool condition, Evaluation *early, Evaluation *late) const {
    if (QUANTIZATION == 0 || !condition) {
      return;
    }
    tables[0].template contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()], early, late);
    tables[1].template contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()], early, late);
    tables[2].template contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()], early, late);
    tables[3].template contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()], early, late);
    tables[4].template contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()], early, late);
    tables[5].template contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()], early, late);
    tables[6].template contribute<US>(pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::PAWN>()], early, late);
    tables[7].template contribute<US>(pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::KNIGHT>()], early, late);
    tables[8].template contribute<US>(pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::BISHOP>()], early, late);
    tables[9].template contribute<US>(pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::ROOK>()], early, late);
    tables[10].template contribute<US>(pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::QUEEN>()], early, late);
    tables[11].template contribute<US>(pos.pieceBitboards_[coloredPiece<opposite_color<US>(), Piece::KING>()], early, late);
  }
};

/**
 * Quantized Square Table Evaluator
 *
 * Uses an un-quantized piece-square table for evaluation, 
 * plus a host of quantized, conditional square tables for finer adjustments.
 */
struct QstEvaluator : public EvaluatorInterface {
  constexpr static size_t QUANTIZATION = 8;
  // Normal piece-square tables.
  ConditionedTaperedPieceSquareTables<QUANTIZATION> pieces;
  
  TaperedQuantizedSquareTable<QUANTIZATION> ourPassedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirPassedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> ourIsolatedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirIsolatedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> ourDoubledPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirDoubledPawns;

  TaperedQuantizedSquareTable<QUANTIZATION> badOurPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> badOurKnights;
  TaperedQuantizedSquareTable<QUANTIZATION> badOurBishops;
  TaperedQuantizedSquareTable<QUANTIZATION> badOurRooks;
  TaperedQuantizedSquareTable<QUANTIZATION> badOurQueens;
  TaperedQuantizedSquareTable<QUANTIZATION> badOurKings;
  TaperedQuantizedSquareTable<QUANTIZATION> badTheirPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> badTheirKnights;
  TaperedQuantizedSquareTable<QUANTIZATION> badTheirBishops;
  TaperedQuantizedSquareTable<QUANTIZATION> badTheirRooks;
  TaperedQuantizedSquareTable<QUANTIZATION> badTheirQueens;
  TaperedQuantizedSquareTable<QUANTIZATION> badTheirKings;

  ConditionedTaperedPieceSquareTables<QUANTIZATION> conditionalOnOurQueen;
  ConditionedTaperedPieceSquareTables<QUANTIZATION> conditionalOnTheirQueen;
  Evaluation bias[2];


  QstEvaluator() {
    std::string filename = "runs/20260201-171038/model.bin";
    std::ifstream in(filename, std::ios::binary);
    this->load(in);
    in.close();
  }

  void load(std::istream& in) {
    pieces.load_table(in, "base_psq");
    ourPassedPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_psd_pwns");
    theirPassedPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_psd_pwns");
    ourIsolatedPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_iso_pwns");
    theirIsolatedPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_iso_pwns");
    ourDoubledPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "our_dbl_pwns");
    theirDoubledPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "tir_dbl_pwns");

    badOurPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_our_p");
    badOurKnights = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_our_n");
    badOurBishops = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_our_b");
    badOurRooks = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_our_r");
    badOurQueens = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_our_q");
    badOurKings = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_our_k");
    badTheirPawns = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_tir_p");
    badTheirKnights = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_tir_n");
    badTheirBishops = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_tir_b");
    badTheirRooks = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_tir_r");
    badTheirQueens = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_tir_q");
    badTheirKings = TaperedQuantizedSquareTable<QUANTIZATION>::load_table(in, "bad_tir_k");

    conditionalOnTheirQueen.load_table(in, "tir_q");
    conditionalOnOurQueen.load_table(in, "our_q");

    float biasWeights[2];
    load_flat<2>(in, biasWeights, "bias");
    bias[0] = static_cast<Evaluation>(std::round(biasWeights[0] * 200.0f));
    bias[1] = static_cast<Evaluation>(std::round(biasWeights[1] * 200.0f));
  }

  // Extract features from the position from mover's perspective.
  template<Color US>
  static void get_features(const Position& pos, std::vector<Bitboard> *out) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
    constexpr Direction kBackward = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;

    const size_t startingLen = out->size();

    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    PawnAnalysis<US> pawnAnalysis(pos);

    out->push_back(pawnAnalysis.ourPassedPawns);
    out->push_back(pawnAnalysis.theirPassedPawns);
    out->push_back(pawnAnalysis.ourIsolatedPawns);
    out->push_back(pawnAnalysis.theirIsolatedPawns);
    out->push_back(pawnAnalysis.ourDoubledPawns);
    out->push_back(pawnAnalysis.theirDoubledPawns);

    Threats<US> threats(pos);
    out->push_back(threats.badForOur[Piece::PAWN]);
    out->push_back(threats.badForOur[Piece::KNIGHT]);
    out->push_back(threats.badForOur[Piece::BISHOP]);
    out->push_back(threats.badForOur[Piece::ROOK]);
    out->push_back(threats.badForOur[Piece::QUEEN]);
    out->push_back(threats.badForOur[Piece::KING]);
    out->push_back(threats.badForTheir[Piece::PAWN]);
    out->push_back(threats.badForTheir[Piece::KNIGHT]);
    out->push_back(threats.badForTheir[Piece::BISHOP]);
    out->push_back(threats.badForTheir[Piece::ROOK]);
    out->push_back(threats.badForTheir[Piece::QUEEN]);
    out->push_back(threats.badForTheir[Piece::KING]);

    // Conditional on them having a queen.
    const bool themHasQueen = pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()] > 0;
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()] * themHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()] * themHasQueen);

    // Conditional on us having a queen.
    const bool usHasQueen = pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()] > 0;
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()] * usHasQueen);
    out->push_back(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()] * usHasQueen);

    // Flip all bitboards if black to move. This helps us encode
    // symmetry and reduce the number of features.
    for (size_t i = startingLen; i < out->size(); ++i) {
      if constexpr (US == Color::BLACK) {
        (*out)[i] = flip_vertically((*out)[i]);
      }
    }
  }

  template<Color US>
  ColoredEvaluation<US> evaluate(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    int32_t stage = earliness(pos);
    Evaluation early = bias[0];
    Evaluation late = bias[1];

    PawnAnalysis<US> pawnAnalysis(pos);

    pieces.template contribute<US>(pos, true, &early, &late);

    ourPassedPawns.contribute<US>(pawnAnalysis.ourPassedPawns, &early, &late);
    theirPassedPawns.contribute<US>(pawnAnalysis.theirPassedPawns, &early, &late);
    ourIsolatedPawns.contribute<US>(pawnAnalysis.ourIsolatedPawns, &early, &late);
    theirIsolatedPawns.contribute<US>(pawnAnalysis.theirIsolatedPawns, &early, &late);
    ourDoubledPawns.contribute<US>(pawnAnalysis.ourDoubledPawns, &early, &late);
    theirDoubledPawns.contribute<US>(pawnAnalysis.theirDoubledPawns, &early, &late);

    Threats<US> threats(pos);
    badOurPawns.contribute<US>(threats.badForOur[Piece::PAWN], &early, &late);
    badOurKnights.contribute<US>(threats.badForOur[Piece::KNIGHT], &early, &late);
    badOurBishops.contribute<US>(threats.badForOur[Piece::BISHOP], &early, &late);
    badOurRooks.contribute<US>(threats.badForOur[Piece::ROOK], &early, &late);
    badOurQueens.contribute<US>(threats.badForOur[Piece::QUEEN], &early, &late);
    badOurKings.contribute<US>(threats.badForOur[Piece::KING], &early, &late);
    badTheirPawns.contribute<US>(threats.badForTheir[Piece::PAWN], &early, &late);
    badTheirKnights.contribute<US>(threats.badForTheir[Piece::KNIGHT], &early, &late);
    badTheirBishops.contribute<US>(threats.badForTheir[Piece::BISHOP], &early, &late);
    badTheirRooks.contribute<US>(threats.badForTheir[Piece::ROOK], &early, &late);
    badTheirQueens.contribute<US>(threats.badForTheir[Piece::QUEEN], &early, &late);
    badTheirKings.contribute<US>(threats.badForTheir[Piece::KING], &early, &late);

    const bool themHasQueen = pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()] > 0;
    conditionalOnTheirQueen.template contribute<US>(pos, themHasQueen, &early, &late);
    const bool usHasQueen = pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()] > 0;
    conditionalOnOurQueen.template contribute<US>(pos, usHasQueen, &early, &late);

    int32_t eval = int32_t(early) * stage + int32_t(late) * (16 - stage);
    return ColoredEvaluation<US>(-eval / 16);
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
