#ifndef SEARCH_H
#define SEARCH_H

#include <atomic>
#include <bit>
#include <memory>
#include <unordered_set>

#include "../game/Position.h"
#include "../game/utils.h"
#include "../game/movegen/movegen.h"
#include "evaluator.h"
#include "ColoredEvaluation.h"

namespace ChessEngine {

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

  if (SEARCH_TYPE == SearchType::ROOT) {
    // If there are permitted moves, filter the move list to only include those moves.
    if (!thread->permittedMoves_.empty()) {
      ExtMove* writePtr = moves;
      for (ExtMove* move = moves; move < end; ++move) {
        if (thread->permittedMoves_.count(move->move) > 0) {
          *writePtr++ = *move;
        }
      }
      end = writePtr;
    }
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
