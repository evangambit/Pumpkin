#ifndef SRC_EVAL_NNUE_NNUEEVALUATIOR_H
#define SRC_EVAL_NNUE_NNUEEVALUATIOR_H

#include <chrono>
#include <iostream>
#include <fstream>

#include "Nnue.h"
#include "Utils.h"

#include "../../game/Position.h"
#include "../Evaluator.h"

using namespace ChessEngine;

namespace NNUE {

struct WDL {
  double win;
  double draw;
  double loss;
};

struct NnueEvaluator : public EvaluatorInterface {
  std::shared_ptr<Nnue> nnue_model;

  NnueEvaluator(std::shared_ptr<Nnue> model) : nnue_model(model) {}

  // Board listener
  void empty() override {
    nnue_model->clear_accumulator();
  }
  void place_piece(ColoredPiece cp, SafeSquare square) override {
    if (cp == ColoredPiece::NO_COLORED_PIECE) {
      return;
    }
    this->place_piece(to_safe_colored_piece(cp), square);
  }
  void remove_piece(ColoredPiece cp, SafeSquare square) override {
    if (cp == ColoredPiece::NO_COLORED_PIECE) {
      return;
    }
    this->remove_piece(to_safe_colored_piece(cp), square);
  }
  void place_piece(SafeColoredPiece cp, SafeSquare square) override {
    nnue_model->increment(feature_index(cp, square));
  }
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {
    nnue_model->decrement(feature_index(cp, square));
  }


  // EvaluatorInterface

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    assert(pos.turn_ == Color::WHITE);
    Evaluation eval = _evaluate(pos);
    return ColoredEvaluation<Color::WHITE>(eval);
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    assert(pos.turn_ == Color::BLACK);
    Evaluation eval = _evaluate(pos);
    return ColoredEvaluation<Color::BLACK>(eval);
  }

  // TODO: get rid of WDL, stop using doubles.
  Evaluation _evaluate(const Position& pos) {
    #ifndef NDEBUG
      Vector<1024> accCopy = nnue_model->whiteAcc;
    #endif

    if (((pos.currentState_.castlingRights & kCastlingRights_WhiteKing) > 0) != nnue_model->x[WHITE_KINGSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_WhiteKing)) {
        nnue_model->increment(WHITE_KINGSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(WHITE_KINGSIDE_CASTLING_RIGHT);
      }
    }
    if (((pos.currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0) != nnue_model->x[WHITE_QUEENSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_WhiteQueen)) {
        nnue_model->increment(WHITE_QUEENSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(WHITE_QUEENSIDE_CASTLING_RIGHT);
      }
    }
    if (((pos.currentState_.castlingRights & kCastlingRights_BlackKing) > 0) != nnue_model->x[BLACK_KINGSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_BlackKing)) {
        nnue_model->increment(BLACK_KINGSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(BLACK_KINGSIDE_CASTLING_RIGHT);
      }
    }
    if (((pos.currentState_.castlingRights & kCastlingRights_BlackQueen) > 0) != nnue_model->x[BLACK_QUEENSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_BlackQueen)) {
        nnue_model->increment(BLACK_QUEENSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(BLACK_QUEENSIDE_CASTLING_RIGHT);
      }
    }

    int16_t *eval = nnue_model->forward(pos.turn_);
    int16_t score = eval[0];

    #ifndef NDEBUG
      nnue_model->compute_acc_from_scratch(pos);
      int16_t score2 = nnue_model->forward(pos.turn_)[0];
      if (score != score2) {
        std::cerr << "NNUE evaluation mismatch: " << score << " vs " << score2 << std::endl;
        accCopy.print_diff(nnue_model->whiteAcc);
        throw std::runtime_error("NNUE evaluation mismatch");
      }
    #endif

    return Evaluation(score);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<NnueEvaluator>(this->nnue_model->clone());
  }
  std::string to_string() const override {
    return "NnueEvaluator";
  }
};

} // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUEEVALUATIOR_H
