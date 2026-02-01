#ifndef SRC_EVAL_QST_QSTEVALUATOR_H
#define SRC_EVAL_QST_QSTEVALUATOR_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <bit>

#include "../../game/Position.h"
#include "../evaluator.h"
#include "../../game/utils.h"

namespace ChessEngine {

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
    for (size_t i = 0; i < QUANTIZATION; ++i) {
      if constexpr (US == Color::BLACK) {
        *eval -= std::popcount(flip_vertically(occupied) & masks[i]) * weights[i];
      } else {
        *eval += std::popcount(occupied & masks[i]) * weights[i];
      }
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

  template<Color US>
  void contribute(Bitboard occupied, Evaluation *early, Evaluation *late) const {
    earlyWeights.template contribute<US>(occupied, early);
    lateWeights.template contribute<US>(occupied, late);
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
  TaperedQuantizedSquareTable<QUANTIZATION> ourPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> ourKnights;
  TaperedQuantizedSquareTable<QUANTIZATION> ourBishops;
  TaperedQuantizedSquareTable<QUANTIZATION> ourRooks;
  TaperedQuantizedSquareTable<QUANTIZATION> ourQueens;
  TaperedQuantizedSquareTable<QUANTIZATION> ourKings;
  TaperedQuantizedSquareTable<QUANTIZATION> theirPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirKnights;
  TaperedQuantizedSquareTable<QUANTIZATION> theirBishops;
  TaperedQuantizedSquareTable<QUANTIZATION> theirRooks;
  TaperedQuantizedSquareTable<QUANTIZATION> theirQueens;
  TaperedQuantizedSquareTable<QUANTIZATION> theirKings;
  
  TaperedQuantizedSquareTable<QUANTIZATION> ourPassedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirPassedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> ourIsolatedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirIsolatedPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> ourDoubledPawns;
  TaperedQuantizedSquareTable<QUANTIZATION> theirDoubledPawns;

  QstEvaluator() {
    std::string filename = "runs/20260201-144401/model.bin";
    std::ifstream in(filename, std::ios::binary);
    this->load(in);
    in.close();
  }

  //   out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  // out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  // out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  // out.write(tensor.tobytes())

  void load64(std::istream& in, float *out, const std::string& name) {
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

  template<size_t QUANTIZATION>
  void load_table(std::istream& in, TaperedQuantizedSquareTable<QUANTIZATION> *out, const std::string& name) {
    float *weights = new float[128];

    load64(in, weights, "e_" + name);
    load64(in, weights + 64, "l_" + name);

    // Scale weights for better quantization.
    for (size_t i = 0; i < 128; ++i) {
      weights[i] *= 200.0f;
    }
    out->earlyWeights = quantize<QUANTIZATION>(weights);
    out->lateWeights = quantize<QUANTIZATION>(weights + 64);
    delete[] weights;
  }

  void load(std::istream& in) {
    load_table(in, &ourPawns, "our_pawns");
    load_table(in, &ourKnights, "our_knights");
    load_table(in, &ourBishops, "our_bishops");
    load_table(in, &ourRooks, "our_rooks");
    load_table(in, &ourQueens, "our_queens");
    load_table(in, &ourKings, "our_king");
    load_table(in, &theirPawns, "their_pawns");
    load_table(in, &theirKnights, "their_knights");
    load_table(in, &theirBishops, "their_bishops");
    load_table(in, &theirRooks, "their_rooks");
    load_table(in, &theirQueens, "their_queens");
    load_table(in, &theirKings, "their_king");
    load_table(in, &ourPassedPawns, "our_psd_pwns");
    load_table(in, &theirPassedPawns, "their_psd_pwns");
    load_table(in, &ourIsolatedPawns, "our_iso_pwns");
    load_table(in, &theirIsolatedPawns, "their_iso_pwns");
    load_table(in, &ourDoubledPawns, "our_dbl_pwns");
    load_table(in, &theirDoubledPawns, "their_dbl_pwns");
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

    // Passed pawns
    Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];

    PawnAnalysis<US> pawnAnalysis(pos);

    out->push_back(pawnAnalysis.ourPassedPawns);
    out->push_back(pawnAnalysis.theirPassedPawns);
    out->push_back(pawnAnalysis.ourIsolatedPawns);
    out->push_back(pawnAnalysis.theirIsolatedPawns);
    out->push_back(pawnAnalysis.ourDoubledPawns);
    out->push_back(pawnAnalysis.theirDoubledPawns);

    // Flip all bitboards if black to move. This helps us encode
    // symmetry and reduce the number of features.
    for (size_t i = startingLen; i < out->size(); ++i) {
      if constexpr (US == Color::BLACK) {
        (*out)[i] = flip_vertically((*out)[i]);
      }
    }
  }

  template<Color US>
  ColoredEvaluation<Color::WHITE> evaluate(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    int32_t stage = earliness(pos);
    Evaluation early = 0;
    Evaluation late = 0;

    PawnAnalysis<US> pawnAnalysis(pos);

    ourPawns.contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()], &early, &late);
    ourKnights.contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()], &early, &late);
    ourBishops.contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()], &early, &late);
    ourRooks.contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()], &early, &late);
    ourQueens.contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()], &early, &late);
    ourKings.contribute<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()], &early, &late);

    theirPawns.contribute<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()], &early, &late);
    theirKnights.contribute<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()], &early, &late);
    theirBishops.contribute<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()], &early, &late);
    theirRooks.contribute<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()], &early, &late);
    theirQueens.contribute<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()], &early, &late);
    theirKings.contribute<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()], &early, &late);

    ourPassedPawns.contribute<US>(pawnAnalysis.ourPassedPawns, &early, &late);
    theirPassedPawns.contribute<US>(pawnAnalysis.theirPassedPawns, &early, &late);
    ourIsolatedPawns.contribute<US>(pawnAnalysis.ourIsolatedPawns, &early, &late);
    theirIsolatedPawns.contribute<US>(pawnAnalysis.theirIsolatedPawns, &early, &late);
    ourDoubledPawns.contribute<US>(pawnAnalysis.ourDoubledPawns, &early, &late);
    theirDoubledPawns.contribute<US>(pawnAnalysis.theirDoubledPawns, &early, &late);

    int32_t eval = early * stage + late * (16 - stage);
    return ColoredEvaluation<Color::WHITE>(eval / 16);
  }

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    return evaluate<Color::WHITE>(pos);
  }

  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    return -evaluate<Color::BLACK>(pos);
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
    return "PieceSquareEvaluator";
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
