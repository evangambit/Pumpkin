#ifndef SEARCH_H
#define SEARCH_H

#include <atomic>
#include <bit>
#include <memory>
#include <unordered_set>

#include "../game/Position.h"
#include "../game/utils.h"

namespace ChessEngine {


template<Color TURN>
struct ColoredEvaluation {
  int16_t value;
  explicit ColoredEvaluation(int16_t v) : value(v) {}
  ColoredEvaluation<opposite_color<TURN>()> operator-() const {
    return ColoredEvaluation<opposite_color<TURN>()>(-value);
  }
  bool operator>=(const ColoredEvaluation<TURN>& other) const {
    return value >= other.value;
  }
  bool operator<=(const ColoredEvaluation<TURN>& other) const {
    return value <= other.value;
  }
  bool operator>(const ColoredEvaluation<TURN>& other) const {
    return value > other.value;
  }
  bool operator<(const ColoredEvaluation<TURN>& other) const {
    return value < other.value;
  }
  bool operator==(const ColoredEvaluation<TURN>& other) const {
    return value == other.value;
  }
  bool operator!=(const ColoredEvaluation<TURN>& other) const {
    return value != other.value;
  }
  ColoredEvaluation<TURN>& operator+=(const ColoredEvaluation<TURN>& other) {
    value += other.value;
    return *this;
  }
  friend std::ostream& operator<<(std::ostream& os, const ColoredEvaluation<TURN>& eval) {
    os << (TURN == Color::WHITE ? eval.value : -eval.value);
    return os;
  }
};


struct EvaluatorInterface : public BoardListener {
  virtual ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) = 0;
  virtual ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) = 0;
  virtual std::shared_ptr<EvaluatorInterface> clone() const = 0;
};

struct SimpleEvaluator : public EvaluatorInterface {
   ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    ColoredEvaluation<Color::WHITE> totalEval(0);
    for (int i = 0; i < ColoredPiece::NUM_COLORED_PIECES; ++i) {
      totalEval = ColoredEvaluation<Color::WHITE>(totalEval.value + kPieceValues[i].value * std::popcount(pos.pieceBitboards_[i]));
    }
    return totalEval;
  }
  
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    return -evaluate_white(pos);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<SimpleEvaluator>();
  }

  inline static ColoredEvaluation<Color::WHITE> kPieceValues[ColoredPiece::NUM_COLORED_PIECES] = {
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
  void remove_piece(SafeSquare square) override {}
};

struct Thread {
  uint64_t id_;
  Position position_;
  std::shared_ptr<EvaluatorInterface> evaluator_;
  std::unordered_set<Move> permittedMoves_;
  uint64_t nodeCount_{0};

  Thread(
    uint64_t id,
    const Position& pos,
    std::shared_ptr<EvaluatorInterface> evaluator,
    const std::unordered_set<Move>& permittedMoves
  )
    : id_(id), position_(pos), evaluator_(evaluator->clone()), permittedMoves_(permittedMoves) {}
  std::atomic<bool> stopSearchFlag{false};
};

template<Color TURN>
struct SearchResult {
  SearchResult() : bestMove(kNullMove), evaluation(0) {}
  SearchResult(Move move, Evaluation eval) : bestMove(move), evaluation(ColoredEvaluation<TURN>(eval)) {}
  Move bestMove;
  ColoredEvaluation<TURN> evaluation;

  SearchResult<opposite_color<TURN>()> operator-() const {
    return SearchResult<opposite_color<TURN>()>(bestMove, -evaluation);
  }
};

template<Color TURN>
ColoredEvaluation<TURN> evaluate(std::shared_ptr<EvaluatorInterface> evaluator, const Position& pos) {
  if constexpr (TURN == Color::WHITE) {
    return ColoredEvaluation<TURN>(evaluator->evaluate_white(pos));
  } else {
    return ColoredEvaluation<TURN>(evaluator->evaluate_black(pos));
  }
}

enum SearchType {
  ROOT,  // Useful for multi-PV searches
  NORMAL_SEARCH,
};

template<Color TURN, SearchType SEARCH_TYPE>
SearchResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta) {
  if (depth == 0) {
    return SearchResult<TURN>(kNullMove, evaluate<TURN>(thread->evaluator_, thread->position_).value);
  }

  ExtMove moves[kMaxNumMoves];
  ExtMove* end;
  if (SEARCH_TYPE == SearchType::ROOT) {
    end = compute_legal_moves<TURN>(&thread->position_, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::ALL_MOVES>(thread->position_, moves);
  }

  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(
    thread->position_,
    lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
  );

  if (moves == end) {
    if (inCheck) {
      return SearchResult<TURN>(kNullMove, kCheckmate);
    } else {
      return SearchResult<TURN>(kNullMove, 0);
    }
  }

  // We need to check this *after* we do the checkmate test above.
  if (thread->position_.is_fifty_move_rule()) {
    return SearchResult<TURN>(kNullMove, 0);  // Draw by fifty-move rule
  }

  SearchResult<TURN> bestResult(kNullMove, kMinEval);
  for (ExtMove* move = moves; move < end; ++move) {
    make_move<TURN>(&thread->position_, move->move);
    ColoredEvaluation<TURN> eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, depth - 1, -beta, -alpha).evaluation;
    undo<TURN>(&thread->position_);
    if (eval.value > bestResult.evaluation.value) {
      bestResult = SearchResult<TURN>(move->move, eval.value);
      if (bestResult.evaluation.value > alpha.value) {
        alpha = ColoredEvaluation<TURN>(bestResult.evaluation.value);
      }
      if (alpha.value >= beta.value) {
        break;
      }
    }
  }

  if (bestResult.evaluation.value < kLongestForcedMate) {
    bestResult.evaluation = ColoredEvaluation<TURN>(bestResult.evaluation.value + 1);
  } else if (bestResult.evaluation.value > -kLongestForcedMate) {
    bestResult.evaluation = ColoredEvaluation<TURN>(bestResult.evaluation.value - 1);
  }

  return bestResult;
}

}  // namespace ChessEngine

#endif  // SEARCH_H
