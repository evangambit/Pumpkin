#ifndef BOARDLISTENER_H
#define BOARDLISTENER_H

#include <cassert>
#include <cstdint>
#include <cstring>

#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "Geometry.h"
#include "Utils.h"
#include "Move.h"
#include "../StringUtils.h"
#include "../eval/ColoredEvaluation.h"

namespace ChessEngine {

struct Position;

struct EvaluatorInterface {
  virtual ~EvaluatorInterface() = default;

  // Evaluation methods
  virtual ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) = 0;
  virtual ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) = 0;
  virtual std::shared_ptr<EvaluatorInterface> clone() const = 0;
  virtual size_t num_features() const { return 0; }
  virtual std::string to_string() const = 0;
  virtual int8_t *write_features(int8_t* features) const {
    return features;
  }

  // BoardListener methods
  virtual void empty() = 0;
  virtual void place_piece(ColoredPiece cp, SafeSquare square) = 0;
  virtual void remove_piece(ColoredPiece cp, SafeSquare square) = 0;
  virtual void place_piece(SafeColoredPiece cp, SafeSquare square) = 0;
  virtual void remove_piece(SafeColoredPiece cp, SafeSquare square) = 0;
};

struct DummyEvaluator : public EvaluatorInterface {

  virtual ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    return ColoredEvaluation<Color::WHITE>(0);
  }
  virtual ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    return ColoredEvaluation<Color::BLACK>(0);
  }
  virtual std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<DummyEvaluator>();
  }
  virtual size_t num_features() const override { return 0; }
  virtual std::string to_string() const override {
    return "DummyEvaluator";
  }

  void empty() override {}
  void place_piece(ColoredPiece cp, SafeSquare square) override {}
  void remove_piece(ColoredPiece cp, SafeSquare square) override {}
  void place_piece(SafeColoredPiece cp, SafeSquare square) override {}
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {}
};

}  // namespace ChessEngine

#endif  // BOARDLISTENER_H
