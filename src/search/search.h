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

#include "transposition_table.h"

namespace ChessEngine {

struct Thread {
  uint64_t id_;
  uint64_t multiPV_;
  Position position_;
  std::shared_ptr<EvaluatorInterface> evaluator_;
  std::unordered_set<Move> permittedMoves_;
  std::vector<std::pair<Move, Evaluation>> primaryVariations_;  // Contains multiPV number of best moves.
  uint64_t nodeCount_{0};

  TranspositionTable* tt_ = nullptr; // Pointer to shared transposition table

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
  NegamaxResult(Move move, ColoredEvaluation<TURN> eval) : bestMove(move), evaluation(eval) {}
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
NegamaxResult<TURN> qsearch(Thread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, int quiescenceDepth) {
  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (quiescenceDepth <= 4) {
    end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(thread->position_, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::CAPTURES>(thread->position_, moves);
  }

  constexpr ColoredPiece enemyKing = coloredPiece<opposite_color<TURN>(), Piece::KING>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(
    thread->position_,
    lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
  );
  if (moves == end && inCheck) {
    return NegamaxResult<TURN>(kNullMove, kCheckmate + plyFromRoot);
  }

  NegamaxResult<TURN> bestResult(kNullMove, evaluate<TURN>(thread->evaluator_, thread->position_));
  if (!inCheck) {
    if (bestResult.evaluation >= beta) {
      return bestResult;
    }
    if (bestResult.evaluation > alpha) {
      alpha = bestResult.evaluation;
    }
  }
  for (ExtMove* move = moves; move < end; ++move) {
    if (thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) {
      // Don't capture the king. TODO: remove this check by fixing move generation.
      continue;
    }
    make_move<TURN>(&thread->position_, move->move);
    ColoredEvaluation<TURN> eval = -qsearch<opposite_color<TURN>()>(thread, -beta, -alpha, plyFromRoot + 1, quiescenceDepth + 1).evaluation;
    undo<TURN>(&thread->position_);
    if (eval > bestResult.evaluation) {
      bestResult.bestMove = move->move;
      bestResult.evaluation = eval;
    }
    if (eval > alpha) {
      alpha = ColoredEvaluation<TURN>(eval.value);
      if (alpha >= beta) {
        break;
      }
    }
  }

  return bestResult;
}

template<Color TURN, SearchType SEARCH_TYPE>
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot) {
  // Transposition Table probe
  if (thread->tt_) {
    TTEntry entry;
    uint64_t key = thread->position_.currentState_.hash;
    if (thread->tt_->probe(key, entry) && entry.depth >= depth) {
      if (entry.bound == BoundType::EXACT) {
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      }
    }
  }
  thread->nodeCount_++;
  if (depth == 0) {
    return qsearch(thread, alpha, beta, plyFromRoot, 0);
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

  constexpr ColoredPiece enemyKing = coloredPiece<opposite_color<TURN>(), Piece::KING>();
  if (moves == end) {
    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    const bool inCheck = can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    );
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
  Move bestMoveTT = kNullMove;
  for (ExtMove* move = moves; move < end; ++move) {
    if (thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) {
      // Don't capture the king. TODO: remove this check by fixing move generation.
      continue;
    }
    make_move<TURN>(&thread->position_, move->move);
    ColoredEvaluation<TURN> eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, depth - 1, -beta, -alpha, plyFromRoot + 1).evaluation;
    undo<TURN>(&thread->position_);
    if (eval > bestResult.evaluation) {
      bestResult.bestMove = move->move;
      bestResult.evaluation = eval;
      bestMoveTT = move->move;
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

  // Store in Transposition Table
  if (thread->tt_) {
    BoundType bound = BoundType::EXACT;
    if (bestResult.evaluation <= alpha) bound = BoundType::UPPER;
    else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;
    thread->tt_->store(
      thread->position_.currentState_.hash,
      bestMoveTT,
      depth,
      bestResult.evaluation.value,
      bound,
      plyFromRoot
    );
  }

  return bestResult;
}

template<Color TURN>
struct SearchResult {
  std::vector<std::pair<Move, ColoredEvaluation<TURN>>> primaryVariations;
  Move bestMove;
  ColoredEvaluation<TURN> evaluation;
  uint64_t nodeCount_;
};

template<Color TURN>
SearchResult<TURN> _search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable *tt) {
  pos.set_listener(evaluator);
  tt->new_search();
  Thread thread(0, pos, evaluator, multiPV, std::unordered_set<Move>());
  thread.tt_ = tt;
  NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(
    &thread, depth,
    /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
    /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
    /*plyFromRoot=*/0
  );
  std::vector<std::pair<Move, ColoredEvaluation<TURN>>> convertedPVs;
  for (const auto& pv : thread.primaryVariations_) {
    convertedPVs.push_back(std::make_pair(pv.first, ColoredEvaluation<TURN>(pv.second)));
  }
  return SearchResult{convertedPVs, result.bestMove, result.evaluation, thread.nodeCount_};
}

inline SearchResult<Color::WHITE> search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable* tt) {
  if (pos.turn_ == Color::WHITE) {
    return _search<Color::WHITE>(pos, evaluator, depth, multiPV, tt);
  } else {
    SearchResult<Color::BLACK> result = _search<Color::BLACK>(pos, evaluator, depth, multiPV, tt);
    std::vector<std::pair<Move, ColoredEvaluation<Color::WHITE>>> convertedVariations;
    for (const auto& pv : result.primaryVariations) {
      convertedVariations.push_back(std::make_pair(pv.first, -pv.second));
    }
    return SearchResult<Color::WHITE>{convertedVariations, result.bestMove, -result.evaluation};
  }
}

}  // namespace ChessEngine

#endif  // SEARCH_H
