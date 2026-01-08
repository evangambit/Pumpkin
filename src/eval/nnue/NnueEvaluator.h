#ifndef SRC_EVAL_NNUE_NNUEEVALUATIOR_H
#define SRC_EVAL_NNUE_NNUEEVALUATIOR_H

#include <iostream>
#include <fstream>

#include "nnue.h"

#include "../../game/Position.h"
#include "../evaluator.h"

using namespace ChessEngine;

namespace NNUE {

struct WDL {
  double win;
  double draw;
  double loss;
};

struct NnueEvaluator : public EvaluatorInterface {
  Nnue *nnue_model;

  NnueEvaluator(Nnue *model) : nnue_model(model) {}

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
    int16_t *eval = nnue_model->forward(pos.turn_);
    return eval[0];
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<NnueEvaluator>(*this);
  }
  std::string to_string() const override {
    return "NnueEvaluator";
  }
};

} // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUEEVALUATIOR_H