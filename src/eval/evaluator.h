#ifndef SRC_SEARCH_EVALUATOR_H
#define SRC_SEARCH_EVALUATOR_H

#include <memory>
#include "../game/Position.h"
#include "../game/utils.h"
#include "../game/BoardListener.h"
#include "ColoredEvaluation.h"

namespace ChessEngine {

struct EvaluatorInterface : public BoardListener {
  virtual ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) = 0;
  virtual ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) = 0;
  virtual std::shared_ptr<EvaluatorInterface> clone() const = 0;
  virtual size_t num_features() const { return 0; }
  virtual std::string to_string() const = 0;
  virtual int8_t *write_features(int8_t* features) const {
    return features;
  }
};

struct SimpleEvaluator : public EvaluatorInterface {
   ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    ColoredEvaluation<Color::WHITE> totalEval(0);
    for (int i = 0; i < kNumColoredPieces; ++i) {
      totalEval = ColoredEvaluation<Color::WHITE>(totalEval.value + kPieceValues[i].value * std::popcount(pos.pieceBitboards_[ColoredPiece(i)]));
    }
    return totalEval;
  }
  
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    return -evaluate_white(pos);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<SimpleEvaluator>();
  }

  std::string to_string() const override {
    return "SimpleEvaluator";
  }

  inline static ColoredEvaluation<Color::WHITE> kPieceValues[kNumColoredPieces] = {
    ColoredEvaluation<Color::WHITE>(0),    // NO_COLORED_PIECE
    ColoredEvaluation<Color::WHITE>(100),  // WHITE_PAWN
    ColoredEvaluation<Color::WHITE>(320),  // WHITE_KNIGHT
    ColoredEvaluation<Color::WHITE>(330),  // WHITE_BISHOP
    ColoredEvaluation<Color::WHITE>(500),  // WHITE_ROOK
    ColoredEvaluation<Color::WHITE>(900),  // WHITE_QUEEN
    ColoredEvaluation<Color::WHITE>(0),    // WHITE_KING
    ColoredEvaluation<Color::WHITE>(-100),  // BLACK_PAWN
    ColoredEvaluation<Color::WHITE>(-320),  // BLACK_KNIGHT
    ColoredEvaluation<Color::WHITE>(-330),  // BLACK_BISHOP
    ColoredEvaluation<Color::WHITE>(-500),  // BLACK_ROOK
    ColoredEvaluation<Color::WHITE>(-900),  // BLACK_QUEEN
    ColoredEvaluation<Color::WHITE>(0)     // BLACK_KING
  };

  void empty() override {}
  void place_piece(ColoredPiece cp, SafeSquare square) override {}
  void remove_piece(ColoredPiece cp, SafeSquare square) override {}
};

} // namespace ChessEngine

#endif // SRC_SEARCH_EVALUATOR_H
