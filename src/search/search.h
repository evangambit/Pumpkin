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
  uint64_t multiPV_;
  Position position_;
  std::shared_ptr<EvaluatorInterface> evaluator_;
  std::unordered_set<Move> permittedMoves_;
  std::vector<std::pair<Move, Evaluation>> primaryVariations_;  // Contains multiPV number of best moves.
  uint64_t nodeCount_{0};

  Thread(
    uint64_t id,
    const Position& pos,
    std::shared_ptr<EvaluatorInterface> evaluator,
    uint64_t multiPV,
    const std::unordered_set<Move>& permittedMoves
  )
    : id_(id), position_(pos), evaluator_(evaluator), permittedMoves_(permittedMoves), multiPV_(multiPV) {}
  std::atomic<bool> stopSearchFlag{false};
};

template<Color TURN>
struct NegamaxResult {
  NegamaxResult() : bestMove(kNullMove), evaluation(0) {}
  NegamaxResult(Move move, Evaluation eval) : bestMove(move), evaluation(ColoredEvaluation<TURN>(eval)) {}
  Move bestMove;
  ColoredEvaluation<TURN> evaluation;

  NegamaxResult<opposite_color<TURN>()> operator-() const {
    return NegamaxResult<opposite_color<TURN>()>(bestMove, -evaluation);
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

template<Color TURN>
NegamaxResult<TURN> qsearch(Thread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot) {
  // TODO: Implement quiescence search
  return NegamaxResult<TURN>(kNullMove, evaluate<TURN>(thread->evaluator_, thread->position_).value);
}

template<Color TURN, SearchType SEARCH_TYPE>
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot) {
  if (depth == 0) {
    return qsearch(thread, alpha, beta, plyFromRoot);
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
      return NegamaxResult<TURN>(kNullMove, kCheckmate + plyFromRoot);
    } else {
      return NegamaxResult<TURN>(kNullMove, 0);
    }
  }

  // We need to check this *after* we do the checkmate test above, since you can win on the 50th move.
  if (thread->position_.is_fifty_move_rule()) {
    return NegamaxResult<TURN>(kNullMove, 0);
  }

  NegamaxResult<TURN> bestResult(kNullMove, kMinEval);
  for (ExtMove* move = moves; move < end; ++move) {
    make_move<TURN>(&thread->position_, move->move);
    ColoredEvaluation<TURN> eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, depth - 1, -beta, -alpha, plyFromRoot + 1).evaluation;
    undo<TURN>(&thread->position_);
    if (eval > bestResult.evaluation) {
      bestResult.bestMove = move->move;
      bestResult.evaluation = eval;
    }
    if (eval > alpha) {
      if (SEARCH_TYPE == SearchType::ROOT && thread->multiPV_ > 1) {
        // In multi-PV search, we want to keep track of multiple best moves and
        // only raise alpha if we have the top N moves.

        // We don't really care about optimizing this too much since it only happens
        // at the root of the search.
        thread->primaryVariations_.push_back(std::make_pair(move->move, eval.value));
        std::sort(
          thread->primaryVariations_.begin(),
          thread->primaryVariations_.end(),
          [](const std::pair<Move, Evaluation>& a, const std::pair<Move, Evaluation>& b) {
            return a.second > b.second;
          }
        );
        if (thread->primaryVariations_.size() >= thread->multiPV_) {
          alpha = ColoredEvaluation<TURN>(thread->primaryVariations_[thread->multiPV_ - 1].second);
          if (thread->primaryVariations_.size() > thread->multiPV_) {
            thread->primaryVariations_.pop_back();
          }
        }
      } else {
        // If we're not at the root, just update alpha.
        alpha = ColoredEvaluation<TURN>(eval.value);
      }
      if (alpha >= beta) {
        break;
      }
    }
  }

  return bestResult;
}

struct SearchResult {
  std::vector<std::pair<Move, ColoredEvaluation<Color::WHITE>>> primaryVariations;
  Move bestMove;
  ColoredEvaluation<Color::WHITE> evaluation;
};

inline SearchResult search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV = 1) {
  pos.set_listener(evaluator);
  Thread thread(0, pos, evaluator, multiPV, std::unordered_set<Move>());
  if (pos.turn_ == Color::WHITE) {
    NegamaxResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(
      &thread, depth,
      /*alpha=*/ColoredEvaluation<Color::WHITE>(kMinEval),
      /*beta=*/ColoredEvaluation<Color::WHITE>(kMaxEval),
      /*plyFromRoot=*/0
    );
    std::vector<std::pair<Move, ColoredEvaluation<Color::WHITE>>> convertedPVs;
    for (const auto& pv : thread.primaryVariations_) {
      convertedPVs.push_back(std::make_pair(pv.first, ColoredEvaluation<Color::WHITE>(pv.second)));
    }
    return SearchResult{convertedPVs, result.bestMove, result.evaluation};
  } else {
    NegamaxResult<Color::BLACK> result = negamax<Color::BLACK, SearchType::ROOT>(
      &thread, depth,
      /*alpha=*/ColoredEvaluation<Color::BLACK>(kMinEval),
      /*beta=*/ColoredEvaluation<Color::BLACK>(kMaxEval),
      /*plyFromRoot=*/0
    );
    std::vector<std::pair<Move, ColoredEvaluation<Color::WHITE>>> convertedPVs;
    for (const auto& pv : thread.primaryVariations_) {
      convertedPVs.push_back(std::make_pair(pv.first, -ColoredEvaluation<Color::BLACK>(pv.second)));
    }
    return SearchResult{convertedPVs, result.bestMove, -result.evaluation};
  }
}

}  // namespace ChessEngine

#endif  // SEARCH_H
