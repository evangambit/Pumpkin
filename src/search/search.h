#ifndef SEARCH_H
#define SEARCH_H

#ifndef IS_PRINT_NODE
#define IS_PRINT_NODE 0
#endif

#ifndef IS_PRINT_QNODE
#define IS_PRINT_QNODE 0
#endif

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
  uint64_t qNodeCount_{0};

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
    qNodeCount_(other.qNodeCount_),
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

  friend std::ostream& operator<<(std::ostream& os, const NegamaxResult<TURN>& result) {
    os << "NegamaxResult(bestMove=" << result.bestMove.uci() << ", evaluation=" << result.evaluation.value << ")";
    return os;
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

const Evaluation kMoveOrderingPieceValue[Piece::NUM_PIECES] = {
  0,    // NO_PIECE
  100,  // PAWN
  320,  // KNIGHT
  330,  // BISHOP
  500,  // ROOK
  900,  // QUEEN
  20000 // KING
};

template<Color TURN>
NegamaxResult<TURN> qsearch(Thread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, int quiescenceDepth) {
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search called: alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " quiescenceDepth=" << quiescenceDepth << " history" << thread->position_.history_ << std::endl;
  }

  thread->nodeCount_++;
  thread->qNodeCount_++;

  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (quiescenceDepth <= 4) {
    end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(thread->position_, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::CAPTURES>(thread->position_, moves);
  }

  if (IS_PRINT_QNODE) {
    std::cout << "  " << thread->position_.fen() << std::endl;
    for (ExtMove* m = moves; m != end; ++m) {
      std::cout << "  QMove: " << m->move.uci() << std::endl;
    }
  }

  constexpr ColoredPiece enemyKing = coloredPiece<opposite_color<TURN>(), Piece::KING>();
  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  const bool inCheck = can_enemy_attack<TURN>(
    thread->position_,
    lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
  );
  if (moves == end && inCheck) {
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Checkmate detected in quiescence search." << std::endl;
    }
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

  // Move ordering: captures that capture higher value pieces first.
  for (ExtMove* move = moves; move < end; ++move) {
    assert(move->move.from < 64);
    assert(move->move.to < 64);
    Piece capturedPiece = cp2p(thread->position_.tiles_[move->move.to]);
    assert(capturedPiece < Piece::NUM_PIECES);
    if (capturedPiece != Piece::NO_PIECE) {
      move->score = kMoveOrderingPieceValue[capturedPiece];
    } else {
      move->score = 0;
    }
  }
  std::sort(
    moves,
    end,
    [](const ExtMove& a, const ExtMove& b) {
      return a.score > b.score;
    }
  );

  for (ExtMove* move = moves; move < end; ++move) {
    if (thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) {
      undo<TURN>(&thread->position_);
      std::cout << "Illegal move generated in qsearch: " << move->move.uci() << std::endl;
      std::cout << thread->position_.fen() << std::endl;
      std::cout << thread->position_ << std::endl;
      exit(1);
    }
    make_move<TURN>(&thread->position_, move->move);

    // Move generation can sometimes generate illegal en passant moves.
    if (move->move.moveType == MoveType::EN_PASSANT) {
      const bool inCheck = can_enemy_attack<TURN>(
        thread->position_,
        lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
      );
      if (inCheck) {
        undo<TURN>(&thread->position_);
        continue;
      }
    }

    // Move generation can sometimes generate illegal en passant moves.
    if (move->move.moveType == MoveType::EN_PASSANT) {
      const bool inCheck = can_enemy_attack<TURN>(
        thread->position_,
        lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
      );
      if (inCheck) {
        undo<TURN>(&thread->position_);
        continue;
      }
    }

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

  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search returning: bestMove=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << std::endl;
  }

  return bestResult;
}

template<Color TURN, SearchType SEARCH_TYPE>
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, std::atomic<bool> *stopThinking) {
  assert(thread->position_.turn_ == TURN);
  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Negamax called: depth=" << depth << " alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " history " << thread->position_.history_ << std::endl;
  }
  const ColoredEvaluation<TURN> originalAlpha = alpha;

  // Transposition Table probe
  TTEntry entry;
  uint64_t key = thread->position_.currentState_.hash;
  if (thread->tt_->probe(key, entry) && entry.depth >= depth) {
    if (SEARCH_TYPE != SearchType::ROOT) {
      if (entry.bound == BoundType::EXACT) {
        if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "TT Hit: EXACT" << std::endl;
        }
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
        if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "TT Hit: LOWER" << std::endl;
        }
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
        if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "TT Hit: UPPER" << std::endl;
        }
        return NegamaxResult<TURN>(entry.bestMove, entry.value);
      }
    }
  } else {
    entry.bestMove = kNullMove;
  }

  if (SEARCH_TYPE != SearchType::ROOT && stopThinking->load()) {
    if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Search stopped externally." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, 0);
  }

  thread->nodeCount_++;
  if (depth == 0) {
    if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Entering quiescence search." << std::endl;
    }
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
      if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Checkmate detected." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, kCheckmate + plyFromRoot);
    } else {
      if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Stalemate detected." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, 0);
    }
  }

  // We need to check this *after* we do the checkmate test above, since you can win on the 50th move.
  if (thread->position_.is_fifty_move_rule()) {
    if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Fifty-move rule draw detected." << std::endl;
    }
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

    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    const bool inCheck = can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    );
    if (inCheck) {
      undo<TURN>(&thread->position_);
      std::cout << "Illegal move generated in quiescence search: " << move->move.uci() << std::endl;
      std::cout << thread->position_.fen() << std::endl;
      std::cout << thread->position_ << std::endl;
      exit(1);
      continue;
    }


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

  if (IS_PRINT_NODE) {
  std::cout << repeat("  ", plyFromRoot) << "Storing in TT: depth=" << depth << " eval=" << bestResult.evaluation.value << " bound=" << static_cast<int>(bound) << std::endl;
  }
  thread->tt_->store(
    thread->position_.currentState_.hash,
    bestMoveTT,
    depth,
    bestResult.evaluation.value,
    bound,
    plyFromRoot
  );

  if (IS_PRINT_NODE) {
  std::cout << repeat("  ", plyFromRoot) << "Negamax returning: bestMove=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << std::endl;
  }


  return bestResult;
}

template<Color TURN>
struct Variation {
  Variation() : evaluation(-999) {}
  Variation(Move move, ColoredEvaluation<TURN> eval) : moves(std::vector<Move>({move})), evaluation(eval) {}
  Variation(const std::vector<Move>& moves, ColoredEvaluation<TURN> eval) : moves(moves), evaluation(eval) {}
  std::vector<Move> moves;
  ColoredEvaluation<TURN> evaluation;
};

template<Color TURN>
struct SearchResult {
  SearchResult() : bestMove(kNullMove), evaluation(0), nodeCount_(0), qNodeCount_(0) {}
  SearchResult(
    const std::vector<Variation<TURN>>& primaryVariations,
    Move bestMove,
    ColoredEvaluation<TURN> evaluation,
    uint64_t nodeCount,
    uint64_t qNodeCount
  )
    : primaryVariations(primaryVariations), bestMove(bestMove), evaluation(evaluation), nodeCount_(nodeCount), qNodeCount_(qNodeCount) {}

  std::vector<Variation<TURN>> primaryVariations;
  Move bestMove;
  ColoredEvaluation<TURN> evaluation;
  uint64_t nodeCount_;
  uint64_t qNodeCount_{0};

  SearchResult<opposite_color<TURN>()> operator-() const {
    SearchResult<opposite_color<TURN>()> result;
    result.bestMove = bestMove;
    result.evaluation = -evaluation;
    result.nodeCount_ = nodeCount_;
    result.qNodeCount_ = qNodeCount_;
    for (const auto& pv : primaryVariations) {
      result.primaryVariations.push_back(Variation<opposite_color<TURN>()>(pv.moves, -pv.evaluation));
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const SearchResult<TURN>& result) {
    os << "SearchResult(bestMove=" << result.bestMove.uci() << ", evaluation=" << result.evaluation.value << ", nodeCount=" << result.nodeCount_ << ", qNodeCount=" << result.qNodeCount_ << ", primaryVariations=[";
    for (const auto& pv : result.primaryVariations) {
      os << "(moves=[";
      for (const auto& move : pv.moves) {
        os << move.uci() << ",";
      }
      os << "], evaluation=" << pv.evaluation.value << "), ";
    }
    os << "])";
    return os;
  }
};

void extract_variation_from_tt(
  const Position& pos, TranspositionTable* tt, std::vector<Move>* movesOut, Move startMove);

template<Color TURN>
SearchResult<TURN> negamax_result_to_search_result(const NegamaxResult<TURN>& result, Thread* thread) {
  std::vector<Variation<TURN>> convertedPVs;
  for (const auto& pv : thread->primaryVariations_) {
    std::vector<Move> moves;
    extract_variation_from_tt(thread->position_, thread->tt_, &moves, pv.first);
    convertedPVs.push_back(Variation<TURN>(moves, ColoredEvaluation<TURN>(pv.second)));
  }
  return SearchResult<TURN>(
    convertedPVs,
    result.bestMove,
    result.evaluation,
    thread->nodeCount_,
    thread->qNodeCount_
  );
}

// Color-templated search function to be used by the UCI interface.
template<Color TURN>
SearchResult<TURN> search(Thread* thread, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<TURN>)> onDepthCompleted) {
  assert(thread->position_.turn_ == TURN);
  NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(
    thread,
    1,
    /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
    /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
    /*plyFromRoot=*/0,
    stopThinking
  );
  if (onDepthCompleted != nullptr) {
    SearchResult<TURN> searchResult = negamax_result_to_search_result<TURN>(result, thread);
    onDepthCompleted(1, searchResult);
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
