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

  uint64_t numIncrements = 0;

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
    for (unsigned i = 0; i < kMaxPlyFromRoot + 1; i++) {
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
    SafeSquare bk = vertically_flip_square(lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[ColoredPiece::BLACK_KING]));

    currentFrame->whiteKingSquare = wk;
    currentFrame->blackKingSquare = bk;
    bool whiteKingMoved = (wk != lastFrame->whiteKingSquare);
    bool blackKingMoved = (bk != lastFrame->blackKingSquare);

    // Flip king so it is always on the left half of the board.
    const bool flipOldWK = (lastFrame->whiteKingSquare % 8 > 3);
    const bool flipOldBK = (lastFrame->blackKingSquare % 8 > 3);
    const bool flipWK = (wk % 8 > 3);
    const bool flipBK = (bk % 8 > 3);
    if (flipWK) {
      wk = horizontally_flip_square(wk);
    }
    if (flipBK) {
      bk = horizontally_flip_square(bk);
    }

    // Index manipulation to convert from 8x8 coordinates to 8x4 coordinates.
    int wk_bucket = kKingBuckets[wk / 8 * 4 + wk % 8];
    int bk_bucket = kKingBuckets[bk / 8 * 4 + bk % 8];

    // TODO: it seems theoretically possible that currentFrame's king squares
    // could be the same as wk/bk, in which case we should probably not
    // reset the accumulators.

    // Initialize accumulators: recompute from scratch if king moved, otherwise copy from last frame
    if (whiteKingMoved) {
      ++numIncrements;
      currentFrame->whiteAcc.setZero();
    } else {
      currentFrame->whiteAcc = lastFrame->whiteAcc;
    }
    if (blackKingMoved) {
      currentFrame->blackAcc.setZero();
    } else {
      currentFrame->blackAcc = lastFrame->blackAcc;
    }

    const ChessEngine::PawnAnalysis<Color::WHITE> pawnAnalysis(pos);
    for (NnueFeatureBitmapType i = static_cast<NnueFeatureBitmapType>(0); i < NF_COUNT; i = static_cast<NnueFeatureBitmapType>(i + 1)) {
      const Bitboard oldBitboard = lastFrame->pieceBitboards[i];
      // Note: we can use flipWK for "oldBitboard" because either the old board's king position
      // is the same as the current king position (in which case the old 'flipWk' is the same as
      // the current 'flipWK'), or the king moved (in which case we ignore the old board's features
      // and compute the new board's features from scratch).
      const Bitboard whiteOldboard = flipOldWK ? flip_horizontally(oldBitboard) : oldBitboard;
      const Bitboard blackOldboard = flipOldBK ? flip_horizontally(oldBitboard) : oldBitboard;
      const Bitboard newBitboard = nnue_feature_to_bitboard(i, pos, threats, pawnAnalysis);
      const Bitboard whiteNewboard = flipWK ? flip_horizontally(newBitboard) : newBitboard;
      const Bitboard blackNewboard = flipBK ? flip_horizontally(newBitboard) : newBitboard;

      if (whiteKingMoved) {
        // Add all current pieces for white side
        Bitboard added = whiteNewboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          ++numIncrements;
          currentFrame->whiteAcc += nnue_model->embWeights[wk_bucket * NNUE_INPUT_DIM + feature_index(i, sq)];
        }
      } else {
        // Incremental update for white side
        Bitboard removed = whiteOldboard & ~whiteNewboard;
        while (removed) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(removed);
          ++numIncrements;
          currentFrame->whiteAcc -= nnue_model->embWeights[wk_bucket * NNUE_INPUT_DIM + feature_index(i, sq)];
        }
        Bitboard added = whiteNewboard & ~whiteOldboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          ++numIncrements;
          currentFrame->whiteAcc += nnue_model->embWeights[wk_bucket * NNUE_INPUT_DIM + feature_index(i, sq)];
        }
      }

      if (blackKingMoved) {
        // Add all current pieces for black side
        Bitboard added = blackNewboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          currentFrame->blackAcc += nnue_model->embWeights[bk_bucket * NNUE_INPUT_DIM + flip_feature_index(feature_index(i, sq))];
        }
      } else {
        // Incremental update for black side
        Bitboard removed = blackOldboard & ~blackNewboard;
        while (removed) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(removed);
          currentFrame->blackAcc -= nnue_model->embWeights[bk_bucket * NNUE_INPUT_DIM + flip_feature_index(feature_index(i, sq))];
        }
        Bitboard added = blackNewboard & ~blackOldboard;
        while (added) {
          const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(added);
          currentFrame->blackAcc += nnue_model->embWeights[bk_bucket * NNUE_INPUT_DIM + flip_feature_index(feature_index(i, sq))];
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
      return "NNUE Evaluator (int16_t) (" + std::to_string(numIncrements) + ")";
    } else {
      return "NNUE Evaluator (float) (" + std::to_string(numIncrements) + ")";
    }
  }
};

} // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUEEVALUATIOR_H
