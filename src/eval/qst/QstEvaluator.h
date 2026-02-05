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

template<Color US>
Bitboard orient(Bitboard b) {
  if constexpr (US == Color::BLACK) {
    return flip_vertically(b);
  } else {
    return b;
  }
}

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
  inline void contribute(Bitboard occupied, Evaluation *eval) const {
    static_assert(SCALE == 1 || SCALE == -1, "SCALE must be 1 or -1");
    Evaluation delta = 0;
    for (size_t i = 0; i < QUANTIZATION; ++i) {
      delta += std::popcount(occupied & masks[i]) * weights[i];
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

/**
  * Two quantized square tables, one for early game and one for late game.
  */
template<size_t QUANTIZATION>
struct TaperedQuantizedSquareTable {
  QuantizedSquareTable<QUANTIZATION> earlyWeights;
  QuantizedSquareTable<QUANTIZATION> lateWeights;

  template<int SCALE = 1>
  void contribute(Bitboard occupied, Evaluation *early, Evaluation *late) const {
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
  // under the influence of enemy pieces, so it's bad for that piece to be there.
  // A square where a rook cannot safely exist is moderately under the influence of enemy pieces.
  // Etc.
  // control[0-5] = bad for our PNBRQK, control[6-11] = bad for their PNBRQK
  TaperedQuantizedSquareTable<4> control[12];

  // Bitboard for each piece type, indicating whether they're in danger of being captured.
  TaperedQuantizedSquareTable<2> capturable[6];

  Evaluation biases[2];

  // Extract features from the position from mover's perspective.
  // This means the mover's pawns are always moving NORTH. This is
  // accomplished by flipping all bitboards vertically if the mover is black
  // (see the end of this function).
  template<Color US>
  void get_features(const Position& pos, std::vector<Bitboard> *out) {
    this->evaluate<US>(pos);
    for (size_t i = 0; i < NUM_FEATURES; ++i) {
      out->push_back(features[i]);
    }
  }

  // It's a little hacky to store features as a member variable, but it helps
  // ensure that get_features and evaluate are consistent with each other.
  static constexpr size_t NUM_FEATURES = 44;
  Bitboard features[NUM_FEATURES];

  template<Color US>
  ColoredEvaluation<US> evaluate(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    int32_t stage = earliness(pos);
    Evaluation early = biases[0];
    Evaluation late = biases[1];

    PawnAnalysis<US> pawnAnalysis(pos);

    features[0] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    features[1] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    pieces[0].contribute(features[0], &early, &late);
    pieces[0].contribute<-1>(flip_vertically(features[1]), &early, &late);

    features[2] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]);
    features[3] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]);
    pieces[1].contribute(features[2], &early, &late);
    pieces[1].contribute<-1>(flip_vertically(features[3]), &early, &late);

    features[4] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    features[5] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    pieces[2].contribute(features[4], &early, &late);
    pieces[2].contribute<-1>(flip_vertically(features[5]), &early, &late);

    features[6] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    features[7] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    pieces[3].contribute(features[6], &early, &late);
    pieces[3].contribute<-1>(flip_vertically(features[7]), &early, &late);

    features[8] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]);
    features[9] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]);
    pieces[4].contribute(features[8], &early, &late);
    pieces[4].contribute<-1>(flip_vertically(features[9]), &early, &late);

    features[10] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    features[11] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);
    pieces[5].contribute(features[10], &early, &late);
    pieces[5].contribute<-1>(flip_vertically(features[11]), &early, &late);

    // kingAssumingNoCastling contributions
    constexpr uint8_t ourKingsideCastling = US == Color::WHITE ? kCastlingRights_WhiteKing : kCastlingRights_BlackKing;
    constexpr uint8_t ourQueensideCastling = US == Color::WHITE ? kCastlingRights_WhiteQueen : kCastlingRights_BlackQueen;
    constexpr uint8_t theirKingsideCastling = US == Color::WHITE ? kCastlingRights_BlackKing : kCastlingRights_WhiteKing;
    constexpr uint8_t theirQueensideCastling = US == Color::WHITE ? kCastlingRights_BlackQueen : kCastlingRights_WhiteQueen;
    const bool weCannotCastle = (pos.currentState_.castlingRights & (ourKingsideCastling | ourQueensideCastling)) == 0;
    const bool theyCannotCastle = (pos.currentState_.castlingRights & (theirKingsideCastling | theirQueensideCastling)) == 0;
    features[12] = orient<US>(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]) * weCannotCastle;
    features[13] = orient<US>(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]) * theyCannotCastle;
    kingAssumingNoCastling.contribute(features[12], &early, &late);
    kingAssumingNoCastling.contribute<-1>(flip_vertically(features[13]), &early, &late);

    features[14] = orient<US>(pawnAnalysis.ourPassedPawns);
    features[15] = orient<US>(pawnAnalysis.theirPassedPawns);
    passedPawns.contribute(features[14], &early, &late);
    passedPawns.contribute<-1>(flip_vertically(features[15]), &early, &late);

    features[16] = orient<US>(pawnAnalysis.ourIsolatedPawns);
    features[17] = orient<US>(pawnAnalysis.theirIsolatedPawns);
    isolatedPawns.contribute(features[16], &early, &late);
    isolatedPawns.contribute<-1>(flip_vertically(features[17]), &early, &late);


    features[18] = orient<US>(pawnAnalysis.ourDoubledPawns);
    features[19] = orient<US>(pawnAnalysis.theirDoubledPawns);
    doubledPawns.contribute(features[18], &early, &late);
    doubledPawns.contribute<-1>(flip_vertically(features[19]), &early, &late);

    Threats<US> threats(pos);
    features[20] = orient<US>(threats.badForOur[Piece::PAWN]);
    features[21] = orient<US>(threats.badForTheir[Piece::PAWN]);
    features[22] = orient<US>(threats.badForOur[Piece::KNIGHT]);
    features[23] = orient<US>(threats.badForTheir[Piece::KNIGHT]);
    features[24] = orient<US>(threats.badForOur[Piece::BISHOP]);
    features[25] = orient<US>(threats.badForTheir[Piece::BISHOP]);
    features[26] = orient<US>(threats.badForOur[Piece::ROOK]);
    features[27] = orient<US>(threats.badForTheir[Piece::ROOK]);
    features[28] = orient<US>(threats.badForOur[Piece::QUEEN]);
    features[29] = orient<US>(threats.badForTheir[Piece::QUEEN]);
    features[30] = orient<US>(threats.badForOur[Piece::KING]);
    features[31] = orient<US>(threats.badForTheir[Piece::KING]);
    control[0].contribute(features[20], &early, &late);
    control[0].contribute<-1>(flip_vertically(features[21]), &early, &late);
    control[1].contribute(features[22], &early, &late);
    control[1].contribute<-1>(flip_vertically(features[23]), &early, &late);
    control[2].contribute(features[24], &early, &late);
    control[2].contribute<-1>(flip_vertically(features[25]), &early, &late);
    control[3].contribute(features[26], &early, &late);
    control[3].contribute<-1>(flip_vertically(features[27]), &early, &late);
    control[4].contribute(features[28], &early, &late);
    control[4].contribute<-1>(flip_vertically(features[29]), &early, &late);
    control[5].contribute(features[30], &early, &late);
    control[5].contribute<-1>(flip_vertically(features[31]), &early, &late);

    // capturable contributions (pieces that are in danger of being captured)
    features[32] = orient<US>(threats.badForOur[Piece::PAWN] & pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()]);
    features[33] = orient<US>(threats.badForTheir[Piece::PAWN] & pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()]);
    features[34] = orient<US>(threats.badForOur[Piece::KNIGHT] & pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]);
    features[35] = orient<US>(threats.badForTheir[Piece::KNIGHT] & pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]);
    features[36] = orient<US>(threats.badForOur[Piece::BISHOP] & pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]);
    features[37] = orient<US>(threats.badForTheir[Piece::BISHOP] & pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]);
    features[38] = orient<US>(threats.badForOur[Piece::ROOK] & pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]);
    features[39] = orient<US>(threats.badForTheir[Piece::ROOK] & pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]);
    features[40] = orient<US>(threats.badForOur[Piece::QUEEN] & pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]);
    features[41] = orient<US>(threats.badForTheir[Piece::QUEEN] & pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]);
    features[42] = orient<US>(threats.badForOur[Piece::KING] & pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    features[43] = orient<US>(threats.badForTheir[Piece::KING] & pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);
    capturable[0].contribute(features[32], &early, &late);
    capturable[0].contribute<-1>(flip_vertically(features[33]), &early, &late);
    capturable[1].contribute(features[34], &early, &late);
    capturable[1].contribute<-1>(flip_vertically(features[35]), &early, &late);
    capturable[2].contribute(features[36], &early, &late);
    capturable[2].contribute<-1>(flip_vertically(features[37]), &early, &late);
    capturable[3].contribute(features[38], &early, &late);
    capturable[3].contribute<-1>(flip_vertically(features[39]), &early, &late);
    capturable[4].contribute(features[40], &early, &late);
    capturable[4].contribute<-1>(flip_vertically(features[41]), &early, &late);
    capturable[5].contribute(features[42], &early, &late);
    capturable[5].contribute<-1>(flip_vertically(features[43]), &early, &late);

    static_assert(NUM_FEATURES == 44, "features size mismatch");

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
