#ifndef SRC_EVAL_NNUE_NNUEEVALUATIOR_H
#define SRC_EVAL_NNUE_NNUEEVALUATIOR_H

#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "Nnue.h"
#include "Utils.h"

#include "../../game/Position.h"
#include "../../game/Threats.h"
#include "../../game/CreateThreats.h"
#include "../Evaluator.h"

using namespace ChessEngine;

namespace NNUE {

inline Bitboard nnue_feature_to_bitboard(NnueFeatureBitmapType feature, const Position& pos, const Threats& threats) {
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
      Bitboard r = pos.pieceBitboards_[ColoredPiece::WHITE_PAWN];
      for (int file = 0; file < 8; file++) {
        const bool noWhitePawnsOnFile = (kFiles[file] & pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]) == kEmptyBitboard;
        r |= bb(56 + file);
      }
      if (pos.currentState_.castlingRights & kCastlingRights_WhiteKing) {
        r |= bb(0);
      }
      if (pos.currentState_.castlingRights & kCastlingRights_WhiteQueen) {
        r |= bb(1);
      }
      return r;
    }
    case NF_WHITE_KNIGHT:
      return pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT];
    case NF_WHITE_BISHOP:
      return pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP];
    case NF_WHITE_ROOK:
      return pos.pieceBitboards_[ColoredPiece::WHITE_ROOK];
    case NF_WHITE_QUEEN:
      return pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN];
    case NF_WHITE_KING:
      return pos.pieceBitboards_[ColoredPiece::WHITE_KING];
    case NF_WHITE_HANGING_PAWNS: 
      return threats.badForCp(ColoredPiece::WHITE_PAWN) & pos.pieceBitboards_[ColoredPiece::WHITE_PAWN];
    case NF_WHITE_HANGING_KNIGHTS:
      return threats.badForCp(ColoredPiece::WHITE_KNIGHT) & pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT];
    case NF_WHITE_HANGING_BISHOPS:
      return threats.badForCp(ColoredPiece::WHITE_BISHOP) & pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP];
    case NF_WHITE_HANGING_ROOKS:
      return threats.badForCp(ColoredPiece::WHITE_ROOK) & pos.pieceBitboards_[ColoredPiece::WHITE_ROOK];
    case NF_WHITE_HANGING_QUEENS:
      return threats.badForCp(ColoredPiece::WHITE_QUEEN) & pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN];
    case NF_WHITE_HANGING_KINGS:
      return threats.badForCp(ColoredPiece::WHITE_KING) & pos.pieceBitboards_[ColoredPiece::WHITE_KING];
    case NF_BLACK_PAWN: {
      Bitboard r = pos.pieceBitboards_[ColoredPiece::BLACK_PAWN];
      for (int file = 0; file < 8; file++) {
        const bool noBlackPawnsOnFile = (kFiles[file] & pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]) == kEmptyBitboard;
        r |= bb(file);
      }
      if (pos.currentState_.castlingRights & kCastlingRights_BlackKing) {
        r |= bb(56);
      }
      if (pos.currentState_.castlingRights & kCastlingRights_BlackQueen) {
        r |= bb(57);
      }
      return r;
    }
    case NF_BLACK_KNIGHT:
      return pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT];
    case NF_BLACK_BISHOP:
      return pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP];
    case NF_BLACK_ROOK:
      return pos.pieceBitboards_[ColoredPiece::BLACK_ROOK];
    case NF_BLACK_QUEEN:
      return pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN];
    case NF_BLACK_KING:
      return pos.pieceBitboards_[ColoredPiece::BLACK_KING];
    case NF_BLACK_HANGING_PAWNS:
      return threats.badForCp(ColoredPiece::BLACK_PAWN) & pos.pieceBitboards_[ColoredPiece::BLACK_PAWN];
    case NF_BLACK_HANGING_KNIGHTS:
      return threats.badForCp(ColoredPiece::BLACK_KNIGHT) & pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT];
    case NF_BLACK_HANGING_BISHOPS:
      return threats.badForCp(ColoredPiece::BLACK_BISHOP) & pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP];
    case NF_BLACK_HANGING_ROOKS:
      return threats.badForCp(ColoredPiece::BLACK_ROOK) & pos.pieceBitboards_[ColoredPiece::BLACK_ROOK];
    case NF_BLACK_HANGING_QUEENS:
      return threats.badForCp(ColoredPiece::BLACK_QUEEN) & pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN];
    case NF_BLACK_HANGING_KINGS:
      return threats.badForCp(ColoredPiece::BLACK_KING) & pos.pieceBitboards_[ColoredPiece::BLACK_KING];
    default:
      std::cerr << "Invalid NnueFeatureBitmapType: " << feature << std::endl;
  }
  return kEmptyBitboard;
}

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
struct NnueEvaluator : public EvaluatorInterface {
  std::shared_ptr<Nnue<T>> nnue_model;

  TypeSafeArray<Bitboard, NF_COUNT, NnueFeatureBitmapType> lastPieceBitboards;

  NnueEvaluator(std::shared_ptr<Nnue<T>> model) : nnue_model(model) {
    this->empty();
  }

  // Board listener
  void empty() override {
    nnue_model->clear_accumulator();
    lastPieceBitboards.fill(kEmptyBitboard);
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

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos, const Threats& threats) override {
    assert(pos.turn_ == Color::WHITE);
    Evaluation eval = _evaluate(pos, threats);
    return ColoredEvaluation<Color::WHITE>(eval);
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos, const Threats& threats) override {
    assert(pos.turn_ == Color::BLACK);
    Evaluation eval = _evaluate(pos, threats);
    return ColoredEvaluation<Color::BLACK>(eval);
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
    nnue_model->compute_acc_from_scratch(pos, threats);
    T score = nnue_model->forward(pos.turn_)[0];
    if (std::is_same<T, int16_t>::value) {
      return Evaluation(score);
    } else {
      const int64_t v = std::round(score * (1 << SCALE_SHIFT));
      const int64_t maxVal = (1 << 15) - 1;
      const int64_t minVal = -(1 << 15);
      return Evaluation(static_cast<int16_t>(std::max(minVal, std::min(maxVal, v))));
    }
  }

  Evaluation _evaluate(const Position& pos, const Threats& threats) {
    if (_is_material_draw(pos)) {
      return Evaluation(0);
    }
    for (NnueFeatureBitmapType i = static_cast<NnueFeatureBitmapType>(0); i < NF_COUNT; i = static_cast<NnueFeatureBitmapType>(i + 1)) {
      const Bitboard oldBitboard = lastPieceBitboards[i];
      const Bitboard newBitboard = nnue_feature_to_bitboard(i, pos, threats);
      Bitboard diff = oldBitboard & ~newBitboard;
      while (diff) {
        const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(diff);
        nnue_model->decrement(feature_index(i, sq));
      }
      diff = ~oldBitboard & newBitboard;
      while (diff) {
        const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(diff);
        nnue_model->increment(feature_index(i, sq));
      }
      lastPieceBitboards[i] = newBitboard;
    }

    T *eval = nnue_model->forward(pos.turn_);
    int16_t score;
    if (std::is_same<T, int16_t>::value) {
      score = eval[0];
    } else {
      const int64_t v = std::round(eval[0] * (1 << SCALE_SHIFT));
      const int64_t maxVal = (1 << 15) - 1;
      const int64_t minVal = -(1 << 15);
      score = static_cast<int16_t>(std::max(minVal, std::min(maxVal, v)));
    }

    #ifndef NDEBUG
      Evaluation score2 = this->from_scratch(pos, threats);
      bool mismatch;
      if (std::is_same<T, int16_t>::value) {
        mismatch = score != score2;
      } else {
        // Allow a small tolerance for floating point differences.
        mismatch = std::abs(score - score2) > 1;
      }
      if (mismatch) {
        std::cerr << "Score mismatch! Incremental: " << score << ", from scratch: " << score2 << std::endl;
        std::cerr << "Position: " << pos.fen() << std::endl;
        exit(1);
      }
    #endif

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
