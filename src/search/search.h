#ifndef SEARCH_H
#define SEARCH_H

#include <atomic>
#include <bit>
#include <functional>
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
  unsigned depth_{1};
  Position position_;
  std::shared_ptr<EvaluatorInterface> evaluator_;
  std::unordered_set<Move> permittedMoves_;
  std::vector<std::pair<Move, Evaluation>> primaryVariations_;  // Contains multiPV number of best moves.
  uint64_t nodeCount_{0};

  // This pointer should be considered non-owning. The TranspositionTable should created and managed elsewhere.
  TranspositionTable* tt_;

  Thread(
    uint64_t id,
    const Position& pos,
    std::shared_ptr<EvaluatorInterface> evaluator,
    uint64_t multiPV,
    const std::unordered_set<Move>& permittedMoves,
    TranspositionTable* tt
  )
    : id_(id), position_(pos), evaluator_(evaluator), permittedMoves_(permittedMoves), multiPV_(multiPV), tt_(tt) {}
  
  Thread(const Thread& other)
  : id_(other.id_),
    multiPV_(other.multiPV_),
    depth_(other.depth_),
    position_(other.position_),
    evaluator_(other.evaluator_),
    permittedMoves_(other.permittedMoves_),
    primaryVariations_(other.primaryVariations_),
    nodeCount_(other.nodeCount_),
    tt_(other.tt_) {
      this->position_.set_listener(this->evaluator_);
    }

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
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, std::atomic<bool> *stopThinking) {
  // Transposition Table probe
  const ColoredEvaluation<TURN> originalAlpha = alpha;
  TTEntry entry;
  uint64_t key = thread->position_.currentState_.hash;
  if (thread->tt_->probe(key, entry) && entry.depth >= depth) {
    // if ROOT && (multiPV > 0 || permittedMoves not empty)
    // we don't want to short-circuit, since we either need to compute
    // multiple best moves, or we need to filter by permitted moves.
    if (SEARCH_TYPE != SearchType::ROOT || (thread->multiPV_ == 1 && thread->permittedMoves_.empty())) {
      if (entry.bound == BoundType::EXACT) {
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      }
    }
  } else {
    entry.bestMove = kNullMove;
  }

  if (stopThinking->load()) {
    return NegamaxResult<TURN>(kNullMove, 0);
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

  // Add score to each move.
  for (ExtMove* move = moves; move < end; ++move) {
    move->score = move->move == entry.bestMove ? 1 : 0;
  }
  std::sort(
    moves,
    end,
    [](const ExtMove& a, const ExtMove& b) {
      return a.score > b.score;
    }
  );

  NegamaxResult<TURN> bestResult(kNullMove, kMinEval);
  Move bestMoveTT = kNullMove;
  for (ExtMove* move = moves; move < end; ++move) {
    if (thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) {
      // Don't capture the king. TODO: remove this check by fixing move generation.
      continue;
    }
    make_move<TURN>(&thread->position_, move->move);
    ColoredEvaluation<TURN> eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, depth - 1, -beta, -alpha, plyFromRoot + 1, stopThinking).evaluation;
    undo<TURN>(&thread->position_);
    if (eval > bestResult.evaluation) {
      bestResult.bestMove = move->move;
      bestResult.evaluation = eval;
      bestMoveTT = move->move;
    }
    if (eval > alpha) {
      if (SEARCH_TYPE == SearchType::ROOT) {
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
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;
  thread->tt_->store(
    thread->position_.currentState_.hash,
    bestMoveTT,
    depth,
    bestResult.evaluation.value,
    bound,
    plyFromRoot
  );

  return bestResult;
}

template<Color TURN>
struct SearchResult {
  SearchResult() : bestMove(kNullMove), evaluation(0), nodeCount_(0) {}
  SearchResult(
    const std::vector<std::pair<Move, ColoredEvaluation<TURN>>>& primaryVariations,
    Move bestMove,
    ColoredEvaluation<TURN> evaluation,
    uint64_t nodeCount
  )
    : primaryVariations(primaryVariations), bestMove(bestMove), evaluation(evaluation), nodeCount_(nodeCount) {}

  std::vector<std::pair<Move, ColoredEvaluation<TURN>>> primaryVariations;
  Move bestMove;
  ColoredEvaluation<TURN> evaluation;
  uint64_t nodeCount_;

  SearchResult<opposite_color<TURN>()> operator-() const {
    SearchResult<opposite_color<TURN>()> result;
    result.bestMove = bestMove;
    result.evaluation = -evaluation;
    result.nodeCount_ = nodeCount_;
    for (const auto& pv : primaryVariations) {
      result.primaryVariations.push_back(std::make_pair(pv.first, -pv.second));
    }
    return result;
  }
};

template<Color TURN>
SearchResult<TURN> negamax_result_to_search_result(const NegamaxResult<TURN>& result, Thread* thread) {
  std::vector<std::pair<Move, ColoredEvaluation<TURN>>> convertedPVs;
  for (const auto& pv : thread->primaryVariations_) {
    convertedPVs.push_back(std::make_pair(pv.first, ColoredEvaluation<TURN>(pv.second)));
  }
  return SearchResult<TURN>(
    convertedPVs,
    result.bestMove,
    result.evaluation,
    thread->nodeCount_
  );
}

// Color-templated search function to be used by the UCI interface.
template<Color TURN>
SearchResult<TURN> search(Thread* thread, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<TURN>)> onDepthCompleted) {
  NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(
    thread,
    1,
    /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
    /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
    /*plyFromRoot=*/0,
    stopThinking
  );
  if (onDepthCompleted != nullptr) {
    onDepthCompleted(1, negamax_result_to_search_result<TURN>(result, thread));
  }
  for (int i = 2; i <= thread->depth_; ++i) {
    thread->primaryVariations_.clear();
    result = negamax<TURN, SearchType::ROOT>(
      thread,
      i,
      /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
      /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
      /*plyFromRoot=*/0,
      stopThinking
    );
    if (onDepthCompleted != nullptr) {
      onDepthCompleted(i, negamax_result_to_search_result<TURN>(result, thread));
    }
  }
  return negamax_result_to_search_result<TURN>(result, thread);
}

// Non-color-templated search function to be used by the UCI interface.
SearchResult<Color::WHITE> colorless_search(Thread* thread, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<Color::WHITE>)> onDepthCompleted);

// Convenience function to search programmatically without needing to specify color or create a thread.
SearchResult<Color::WHITE> search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable* tt);

}  // namespace ChessEngine

#endif  // SEARCH_H
