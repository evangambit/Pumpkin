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
    WDL wdl = _evaluate(pos);
    double score = wdl.win - wdl.loss;
    return ColoredEvaluation<Color::WHITE>(int32_t(std::min<int16_t>(std::max<int16_t>(-1000, score * 1000), 1000)));
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    assert(pos.turn_ == Color::BLACK);
    WDL wdl = _evaluate(pos);
    double score = wdl.win - wdl.loss;
    return ColoredEvaluation<Color::BLACK>(int32_t(std::min<int16_t>(std::max<int16_t>(-1000, score * 1000), 1000)));
  }

  // TODO: get rid of WDL, stop using doubles.
  WDL _evaluate(const Position& pos) {
    int16_t *eval = nnue_model->forward(pos.turn_);
    int earliness = 0;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]) * 3;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]) * 3;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]) * 1;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]) * 1;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]) * 1;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]) * 1;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]) * 1;
    earliness += std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]) * 1;
    earliness = std::min(earliness, 18);
    double early_win = double(eval[0]) / 1000;
    double early_draw = double(eval[1]) / 1000;
    double early_loss = double(eval[2]) / 1000;
    double late_win = double(eval[3]) / 1000;
    double late_draw = double(eval[4]) / 1000;
    double late_loss = double(eval[5]) / 1000;

    double earlyShift = std::max({early_win, early_draw, early_loss});
    double lateShift = std::max({late_win, late_draw, late_loss});
    early_win = std::exp(early_win - earlyShift);
    early_draw = std::exp(early_draw - earlyShift);
    early_loss = std::exp(early_loss - earlyShift);
    late_win = std::exp(late_win - lateShift);
    late_draw = std::exp(late_draw - lateShift);
    late_loss = std::exp(late_loss - lateShift);

    // Softmax
    double earlySum = early_win + early_draw + early_loss;
    double lateSum = late_win + late_draw + late_loss;
    early_win /= earlySum;
    early_draw /= earlySum;
    early_loss /= earlySum;
    late_win /= lateSum;
    late_draw /= lateSum;
    late_loss /= lateSum;

    double win = (early_win * (18 - earliness) + late_win * earliness) / 18.0;
    double draw = (early_draw * (18 - earliness) + late_draw * earliness) / 18.0;
    double loss = (early_loss * (18 - earliness) + late_loss * earliness) / 18.0;

    std::cout << "Win: " << win << " Draw: " << draw << " Loss: " << loss << std::endl;

    return WDL{win, draw, loss};
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