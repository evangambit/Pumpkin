#ifndef SRC_EVAL_NNUE_NNUEEVALUATIOR_H
#define SRC_EVAL_NNUE_NNUEEVALUATIOR_H

#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "Nnue.h"
#include "NnueFeatureBitmapType.h"
#include "Utils.h"

#include "../../game/Position.h"
#include "../../game/Threats.h"
#include "../../game/CreateThreats.h"
#include "../../TypeSafeArray.h"
#include "../Evaluator.h"

using namespace ChessEngine;

namespace NNUE {

inline std::string diff_bstr(Bitboard oldb, Bitboard newb) {
  std::string result;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      int i = y * 8 + x;
      char c;
      if ((oldb & bb(i)) && !(newb & bb(i))) {
        c = '-';
      } else if (!(oldb & bb(i)) && (newb & bb(i))) {
        c = '+';
      } else {
        c = '.';
      }
      result += c;
    }
    result += '\n';
  }
  return result;
}

template<typename T>
struct Frame {
  TypeSafeArray<Bitboard, NF_COUNT, NnueFeatureBitmapType> pieceBitboards;
  Vector<EMBEDDING_DIM, T> whiteAcc;
  Vector<EMBEDDING_DIM, T> blackAcc;
  SafeSquare whiteKingSquare;
  SafeSquare blackKingSquare;
};

template<typename T>
struct NnueEvaluator : public EvaluatorInterface {
  std::shared_ptr<Nnue<T>> nnue_model;

  Frame<T> *frames;

  NnueEvaluator(std::shared_ptr<Nnue<T>> model) : nnue_model(model) {
    // Add buffer at the beginning.
    frames = new Frame<T>[kMaxPlyFromRoot + 1];
    this->empty();
  }

  ~NnueEvaluator() {
    delete[] frames;
  }

  // Board listener
  void empty() override {
    for (int i = 0; i < kMaxPlyFromRoot + 1; i++) {
      frames[i].pieceBitboards.fill(kEmptyBitboard);
      frames[i].whiteAcc.setZero();
      frames[i].blackAcc.setZero();
      frames[i].whiteKingSquare = SafeSquare(0);
      frames[i].blackKingSquare = SafeSquare(0);
    }
  }
  void place_piece(ColoredPiece cp, SafeSquare square) override {
  }
  void remove_piece(ColoredPiece cp, SafeSquare square) override {
  }
  void place_piece(SafeColoredPiece cp, SafeSquare square) override {
  }
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {
  }


  // EvaluatorInterface

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::WHITE> alpha, ColoredEvaluation<Color::WHITE> beta) override {
    assert(pos.turn_ == Color::WHITE);
    Evaluation eval = _evaluate(pos, threats, plyFromRoot, true, alpha.value, beta.value);
    return ColoredEvaluation<Color::WHITE>(eval);
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::BLACK> alpha, ColoredEvaluation<Color::BLACK> beta) override {
    assert(pos.turn_ == Color::BLACK);
    Evaluation eval = _evaluate(pos, threats, plyFromRoot, true, alpha.value, beta.value);
    return ColoredEvaluation<Color::BLACK>(eval);
  }
  
  void update_accumulator(const Position& pos, const Threats& threats, int plyFromRoot) override {
    _evaluate(pos, threats, plyFromRoot, false, kMinEval, kMaxEval);
  }

  bool _is_material_draw(const Position& pos) const {
    if ((pos.pieceBitboards_[ColoredPiece::WHITE_PAWN] | pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]) == kEmptyBitboard) {
      int whiteMinor = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
      int blackMinor = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT] | pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
      int whiteMajor = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK] | pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
      int blackMajor = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK] | pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);
      return whiteMajor + blackMajor == 0 && whiteMinor <= 1 && blackMinor <= 1;
    }
    return false;
  }

  // Very slow, but useful for testing (to ensure that incremental updates are correct).
  Evaluation from_scratch(const Position& pos, const Threats& threats) const {
    ChessEngine::SafeSquare whiteKingSquare = ChessEngine::lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ChessEngine::ColoredPiece::WHITE_KING]);
    ChessEngine::SafeSquare blackKingSquare = ChessEngine::lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ChessEngine::ColoredPiece::BLACK_KING]);
    Features features(whiteKingSquare, blackKingSquare);
    for (NnueFeatureBitmapType i = NnueFeatureBitmapType(0); i < NF_COUNT; i = NnueFeatureBitmapType(i + 1)) {
      Bitboard bb = nnue_feature_to_bitboard(i, pos, threats);
      while (bb) {
        unsigned sq = pop_lsb_i_promise_board_is_not_empty(bb);
        features.addFeature(feature_index(i, sq));
      }
    }
    
    Vector<EMBEDDING_DIM, T> whiteAcc;
    Vector<EMBEDDING_DIM, T> blackAcc;
    whiteAcc.setZero();
    blackAcc.setZero();
    for (int i = 0; i < features.size(); i++) {
      int16_t pieceIdx = features[i];
      size_t whiteIdx = whiteKingSquare * NNUE_INPUT_DIM + pieceIdx;
      size_t blackIdx = vertically_flip_square(blackKingSquare) * NNUE_INPUT_DIM + flip_feature_index(pieceIdx);
      nnue_model->increment(&whiteAcc, whiteIdx, &blackAcc, blackIdx);
    }
    T score;
    if (pos.turn_ == Color::WHITE) {
      score = nnue_model->forward(whiteAcc, blackAcc)[0];
    } else {
      score = nnue_model->forward(blackAcc, whiteAcc)[0];
    }
    if (std::is_same<T, int16_t>::value) {
      return Evaluation(score);
    } else {
      const int64_t v = std::round(score * (1 << SCALE_SHIFT));
      const int64_t maxVal = (1 << 15) - 1;
      const int64_t minVal = -(1 << 15);
      return Evaluation(static_cast<int16_t>(std::max(minVal, std::min(maxVal, v))));
    }
  }

  Evaluation _evaluate(const Position& pos, const Threats& threats, int plyFromRoot, bool compute_score, Evaluation alpha, Evaluation beta) {
    if (_is_material_draw(pos)) {
      if (compute_score) {
        return Evaluation(0);
      }
    }
    Frame<T> *lastFrame = frames + plyFromRoot;
    Frame<T> *currentFrame = frames + plyFromRoot + 1;

    // Get current king positions
    SafeSquare wk = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ColoredPiece::WHITE_KING]);
    SafeSquare bk = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ColoredPiece::BLACK_KING]);
    SafeSquare flipped_bk = vertically_flip_square(bk);
    bool whiteKingMoved = (wk != lastFrame->whiteKingSquare);
    bool blackKingMoved = (bk != lastFrame->blackKingSquare);
    currentFrame->whiteKingSquare = wk;
    currentFrame->blackKingSquare = bk;

    // Initialize accumulators: recompute from scratch if king moved, otherwise copy from last frame
    if (whiteKingMoved) {
      currentFrame->whiteAcc.setZero();
    } else {
      currentFrame->whiteAcc = lastFrame->whiteAcc;
    }
    if (blackKingMoved) {
      currentFrame->blackAcc.setZero();
    } else {
      currentFrame->blackAcc = lastFrame->blackAcc;
    }

    for (NnueFeatureBitmapType i = static_cast<NnueFeatureBitmapType>(0); i < NF_COUNT; i = static_cast<NnueFeatureBitmapType>(i + 1)) {
      const Bitboard oldBitboard = lastFrame->pieceBitboards[i];
      const Bitboard newBitboard = nnue_feature_to_bitboard(i, pos, threats);

      if (whiteKingMoved) {
        // Add all current pieces for white side
        Bitboard added = newBitboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          currentFrame->whiteAcc += nnue_model->embWeights[wk * NNUE_INPUT_DIM + feature_index(i, sq)];
        }
      } else {
        // Incremental update for white side
        Bitboard removed = oldBitboard & ~newBitboard;
        while (removed) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(removed);
          currentFrame->whiteAcc -= nnue_model->embWeights[wk * NNUE_INPUT_DIM + feature_index(i, sq)];
        }
        Bitboard added = newBitboard & ~oldBitboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          currentFrame->whiteAcc += nnue_model->embWeights[wk * NNUE_INPUT_DIM + feature_index(i, sq)];
        }
      }

      if (blackKingMoved) {
        // Add all current pieces for black side
        Bitboard added = newBitboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          currentFrame->blackAcc += nnue_model->embWeights[flipped_bk * NNUE_INPUT_DIM + flip_feature_index(feature_index(i, sq))];
        }
      } else {
        // Incremental update for black side
        Bitboard removed = oldBitboard & ~newBitboard;
        while (removed) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(removed);
          currentFrame->blackAcc -= nnue_model->embWeights[flipped_bk * NNUE_INPUT_DIM + flip_feature_index(feature_index(i, sq))];
        }
        Bitboard added = newBitboard & ~oldBitboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          currentFrame->blackAcc += nnue_model->embWeights[flipped_bk * NNUE_INPUT_DIM + flip_feature_index(feature_index(i, sq))];
        }
      }

      currentFrame->pieceBitboards[i] = newBitboard;
    }

    if (!compute_score) {
      return Evaluation(0);
    }

    T *eval;
    const Vector<EMBEDDING_DIM, T>& whiteAcc = currentFrame->whiteAcc;
    const Vector<EMBEDDING_DIM, T>& blackAcc = currentFrame->blackAcc;
    if (pos.turn_ == Color::WHITE) {
      eval = nnue_model->forward(whiteAcc, blackAcc);
    } else {
      eval = nnue_model->forward(blackAcc, whiteAcc);
    }
    int16_t score;
    if (std::is_same<T, int16_t>::value) {
      score = eval[0];
    } else {
      const int64_t v = std::round(eval[0] * (1 << SCALE_SHIFT));
      const int64_t maxVal = (1 << 15) - 1;
      const int64_t minVal = -(1 << 15);
      score = static_cast<int16_t>(std::max(minVal, std::min(maxVal, v)));
    }

    return Evaluation(score);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<NnueEvaluator>(this->nnue_model->clone());
  }
  std::string to_string() const override {
    if (std::is_same<T, int16_t>::value) {
      return "NNUE Evaluator (int16_t)";
    } else {
      return "NNUE Evaluator (float)";
    }
  }
};

} // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUEEVALUATIOR_H
