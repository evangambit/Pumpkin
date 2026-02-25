#ifndef SRC_SEARCH_PIECE_SQUARE_EVALUATOR_H
#define SRC_SEARCH_PIECE_SQUARE_EVALUATOR_H

#include <memory>
#include "../../game/Position.h"
#include "../../game/Utils.h"
#include "../../game/BoardListener.h"
#include "../Evaluator.h"
#include "../ColoredEvaluation.h"

namespace ChessEngine {

struct PieceSquareEvaluator : public EvaluatorInterface {
  Evaluation early;
  Evaluation late;
  static const Evaluation kDefaultPieceSquareTables[2 * 7 * 64];

  PieceSquareEvaluator() : early(0), late(0) {}

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos, const Threats& threats, int plyFromRoot) override {
    int32_t stage = earliness(pos);
    int32_t eval = early * stage + late * (16 - stage);
    return ColoredEvaluation<Color::WHITE>(eval / 16);
  }

  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos, const Threats& threats, int plyFromRoot) override {
    return -evaluate_white(pos, threats, plyFromRoot);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<PieceSquareEvaluator>();
  }

  void empty() override {
    early = 0;
    late = 0;
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
    const size_t index = index_for(cp, square);
    if (cp2color(cp) == Color::WHITE) {
        early += kDefaultPieceSquareTables[index];
        late += kDefaultPieceSquareTables[index + 7 * 64];
    } else {
        early -= kDefaultPieceSquareTables[index];
        late -= kDefaultPieceSquareTables[index + 7 * 64];
    }
  }
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {
    const size_t index = index_for(cp, square);
    if (cp2color(cp) == Color::WHITE) {
        early -= kDefaultPieceSquareTables[index];
        late -= kDefaultPieceSquareTables[index + 7 * 64];
    } else {
        early += kDefaultPieceSquareTables[index];
        late += kDefaultPieceSquareTables[index + 7 * 64];
    }
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

  inline size_t index_for(SafeColoredPiece cp, SafeSquare square) const {
    Piece piece = cp2p(cp);
    Color color = cp2color(cp);
    const int y = color == Color::WHITE ? square / 8 : 7 - (square / 8);
    const int x = square % 8;
    return (piece * 64) + (y * 8) + x;
  }
};

} // namespace ChessEngine

#endif // SRC_SEARCH_PIECE_SQUARE_EVALUATOR_H
