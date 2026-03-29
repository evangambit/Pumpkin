/**
 * This file contains the core search logit, but may not be the most
 * usable API. See search.h for a more user-friendly search interface.
 */

#ifndef NEGAMAX_H
#define NEGAMAX_H

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <unordered_set>

#include "../game/Position.h"
#include "../game/Utils.h"
#include "../game/movegen/movegen.h"
#include "../game/Threats.h"
#include "../game/CreateThreats.h"
#include "../eval/Evaluator.h"
#include "../eval/ColoredEvaluation.h"

#include "transposition_table.h"


// IS_PRINT_NODE and IS_PRINT_QNODE can be used to print debugging information for specific nodes.

#ifndef IS_PRINT_NODE
#define IS_PRINT_NODE 0
// #define IS_PRINT_NODE (thread->position_.currentState_.hash == 8614104399537843458ULL || thread->position_.currentState_.hash == 14669261985347518465ULL)
// #define IS_PRINT_NODE (frame->hash == hash=17514753775184410351ULL && SEARCH_TYPE == SearchType::NORMAL_SEARCH)
// #define IS_PRINT_NODE (plyFromRoot == 0 || frame->hash == 15932567610229845462ULL || frame->hash == 15427882709703266013ULL)
// #define IS_PRINT_NODE (thread->position_.currentState_.hash == 412260009870427727ULL)
// #define IS_PRINT_NODE (thread->frames_[0].hash == 383495967171122001ULL && plyFromRoot <= 2)
#endif

#ifndef IS_PRINT_QNODE
#define IS_PRINT_QNODE 0
// #define IS_PRINT_QNODE ((frame - quiescenceDepth)->hash == 17514877330620511575ULL)
#endif

// If EVAL_AGNOSTIC, we disable optimizations that require tuning for
// the evaluation. This makes comparing evaluators more fair.
#ifndef EVAL_AGNOSTIC
#define EVAL_AGNOSTIC 0
#endif

namespace ChessEngine {

/**
 * Initially we just alternated between replacing the first and second
 * indices when add was called. Now we use an alternative which seems
 * to perform better: moves[0] is considered the "primary" killer move,
 * and moves[1] is considered secondary. When a new killer move is added,
 * it either promotes to primary (if it matches the current secondary) or
 * replaces the secondary.
 */
struct Killers {
  Move moves[2];
  void add(Move move) {
    if (moves[1] == move) {
      std::swap(moves[0], moves[1]);
    } else if (moves[0] != move) {
      moves[1] = move;
    }
  }
  bool contains(Move move) const {
    return moves[0] == move || moves[1] == move;
  }
};

/** Ply-specific information. */
struct Frame {
  Killers killers;
  Move responseTo[Piece::NUM_PIECES][64];
  Move responseFrom[Piece::NUM_PIECES][64];
  uint64_t hash;
  bool inCheck;
  Evaluation staticEval;
};

/**
  * Thread-specific information. e.g. every thread has its own nodeCount_, position, etc.
  */
struct Thread {
  uint64_t id_;
  uint64_t multiPV_;
  unsigned depth_{1};
  std::chrono::high_resolution_clock::time_point stopTime_;
  Position position_;  // Note: position contains a pointer to the evaluator.
  std::unordered_set<Move> permittedMoves_;
  std::vector<std::pair<Move, Evaluation>> primaryVariations_;  // Contains multiPV number of best moves.
  uint64_t nodeCount_{0};
  uint64_t qNodeCount_{0};
  uint64_t nodeLimit_{(uint64_t)-1};
  Frame buffer[4];  // Empty buffer so that frames_[plyFromRoot - 4] is valid.
  Frame frames_[kMaxPlyFromRoot];
  int32_t quietHistory_[Piece::NUM_PIECES][64];
  int32_t captureHistory_[Piece::NUM_PIECES][Piece::NUM_PIECES][64];

  // This pointer should be considered non-owning. The TranspositionTable should created and
  // managed elsewhere since it should be shared across threads and searches.
  TranspositionTable* tt_;

  Thread(
    uint64_t id,
    const Position& pos,
    uint64_t multiPV,
    const std::unordered_set<Move>& permittedMoves,
    TranspositionTable* tt
  ) : id_(id), multiPV_(multiPV), position_(pos), permittedMoves_(permittedMoves), tt_(tt) {
    std::memset(buffer, 0, sizeof(buffer));
    std::memset(frames_, 0, sizeof(frames_));
    std::memset(quietHistory_, 0, sizeof(quietHistory_));
    std::memset(captureHistory_, 0, sizeof(captureHistory_));
  }
  
  Thread(const Thread& other)
  : id_(other.id_),
    multiPV_(other.multiPV_),
    depth_(other.depth_),
    stopTime_(other.stopTime_),
    position_(other.position_),
    permittedMoves_(other.permittedMoves_),
    primaryVariations_(other.primaryVariations_),
    nodeCount_(other.nodeCount_),
    qNodeCount_(other.qNodeCount_),
    nodeLimit_(other.nodeLimit_),
    tt_(other.tt_) {
      std::memcpy(buffer, other.buffer, sizeof(buffer));
      std::memcpy(frames_, other.frames_, sizeof(frames_));
      std::memcpy(quietHistory_, other.quietHistory_, sizeof(quietHistory_));
      std::memcpy(captureHistory_, other.captureHistory_, sizeof(captureHistory_));
    }

  // TODO: when we add multi-threading, we should share stopSearchFlag across threads.
  std::atomic<bool> stopSearchFlag{false};
};

template<Color TURN>
struct NegamaxResult {
  NegamaxResult() : bestMove(kNullMove), evaluation(0) {}
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
ColoredEvaluation<TURN> evaluate(std::shared_ptr<EvaluatorInterface> evaluator, const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta) {
  if constexpr (TURN == Color::WHITE) {
    return evaluator->evaluate_white(pos, threats, plyFromRoot, alpha, beta);
  } else {
    return evaluator->evaluate_black(pos, threats, plyFromRoot, alpha, beta);
  }
}

enum SearchType {
  ROOT,  // Useful for multi-PV searches
  NORMAL_SEARCH,
  NULL_WINDOW_SEARCH,
};

constexpr Evaluation kQMoveOrderingPieceValue[Piece::NUM_PIECES] = {
  1000,    // NO_PIECE (means it's a check)
  100,  // PAWN
  320,  // KNIGHT
  330,  // BISHOP
  500,  // ROOK
  900,  // QUEEN
  20000 // KING
};

// kCheckmate -> -kCheckmate - 1
template<Color TURN>
ColoredEvaluation<opposite_color<TURN>()> to_parent_eval(ColoredEvaluation<TURN> childEval) {
  if (childEval.value < kLongestForcedMate) {
    return ColoredEvaluation<opposite_color<TURN>()>(-childEval.value - 1);
  } else if (childEval.value > -kLongestForcedMate) {
    return ColoredEvaluation<opposite_color<TURN>()>(-childEval.value + 1);
  }
  return -childEval;
}

template<Color TURN>
ColoredEvaluation<opposite_color<TURN>()> to_child_eval(ColoredEvaluation<TURN> parentEval) {
  if (parentEval.value < kLongestForcedMate && parentEval.value > kCheckmate) {
    return ColoredEvaluation<opposite_color<TURN>()>(-parentEval.value + 1);
  } else if (parentEval.value > -kLongestForcedMate && parentEval.value < -kCheckmate) {
    return ColoredEvaluation<opposite_color<TURN>()>(-parentEval.value - 1);
  }
  return -parentEval;
}

template<Color TURN>
NegamaxResult<TURN> qsearch(Thread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, int quiescenceDepth, Frame *frame, std::atomic<bool> *stopThinking) {
  frame->hash = thread->position_.currentState_.hash;
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search called: alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " quiescenceDepth=" << quiescenceDepth << " history" << thread->position_.history_ << std::endl;
  }

  // This can happen when we've already found a checkmate in a previous sibling/ancestor node.
  if (alpha >= beta) {
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Alpha-beta window is invalid (alpha >= beta). Returning beta." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, beta);
  }

  // Prevent stack overflow - return static eval if we've gone too deep
  if (quiescenceDepth >= kMaxQuiescenceDepth || (frame - thread->frames_) >= kMaxPlyFromRoot - 1) {
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Max quiescence depth or ply limit reached, returning static evaluation." << std::endl;
    }
    Threats threats;
    create_threats(thread->position_.pieceBitboards_, thread->position_.colorBitboards_, &threats);
    return NegamaxResult<TURN>(kNullMove, evaluate<TURN>(thread->position_.evaluator_, thread->position_, threats, plyFromRoot, alpha, beta).clamp_(alpha, beta));
  }

  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  frame->inCheck = can_enemy_attack<TURN>(
    thread->position_,
    lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
  );

  // Check if draw by repetition. is_3fold_repetition short-circuits when the
  // last move was a capture or pawn move, so this is near-free in the common
  // qsearch case (captures). It still correctly detects repetitions through
  // check sequences.
  if (thread->position_.is_3fold_repetition(plyFromRoot)) {
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(alpha, beta));
  }

  // Transposition Table probe
  TTEntry entry;
  uint64_t key = thread->position_.currentState_.hash;
  if (thread->tt_->probe(key, entry)) {
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "qTT hit: move=" << entry.bestMove.uci() << " value=" << entry.value << " depth=" << entry.depth << " bound=" << bound_type_to_string(entry.bound) << " hash=" << key << std::endl;
    }
    if (entry.bound == BoundType::EXACT) {
      return NegamaxResult<TURN>(entry.bestMove, ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta));
    } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
      return NegamaxResult<TURN>(entry.bestMove, beta);
    } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
      return NegamaxResult<TURN>(entry.bestMove, alpha);
    }
  } else {
    entry.bestMove = kNullMove;
  }

  const ColoredEvaluation<TURN> originalAlpha = alpha;

  thread->nodeCount_++;
  thread->qNodeCount_++;

  if ((thread->nodeCount_ & 1023) == 0) {
    if (std::chrono::high_resolution_clock::now() >= thread->stopTime_ || thread->nodeCount_ >= thread->nodeLimit_) {
      stopThinking->store(true);
    }
  }

  if (stopThinking->load()) {
    return NegamaxResult<TURN>(kNullMove, originalAlpha);
  }

  ExtMove moves[kMaxNumMoves];
  ExtMove *end;
  if (quiescenceDepth <= 4) {
    end = compute_moves<TURN, MoveGenType::CHECKS_AND_CAPTURES>(thread->position_, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::CAPTURES>(thread->position_, moves);
  }

  // Validate move count is within bounds
  assert(end >= moves && end <= moves + kMaxNumMoves);

  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "In check: " << frame->inCheck << "; numMoves = " << (end - moves) << std::endl;
  }
  if (moves == end && frame->inCheck) {
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Checkmate detected in quiescence search." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate).clamp_(originalAlpha, beta));
  }
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Comparing static evaluation to alpha/beta" << std::endl;
  }

  Threats threats;
  create_threats(thread->position_.pieceBitboards_, thread->position_.colorBitboards_, &threats);
  NegamaxResult<TURN> bestResult(kNullMove, evaluate<TURN>(thread->position_.evaluator_, thread->position_, threats, plyFromRoot, alpha, beta).clamp_(alpha, beta));
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Static evaluation: " << bestResult.evaluation.value << " (hash = " << frame->hash << ")" << std::endl;
  }
  if (!frame->inCheck) {
    if (bestResult.evaluation >= beta) {
      bestResult.evaluation = beta;
      return bestResult;
    }
    if (bestResult.evaluation > alpha) {
      alpha = bestResult.evaluation;
    }
  }

  // Move ordering: captures that capture higher value pieces first.
  assert(!thread->position_.history_.empty() && "qsearch requires history to have at least one move");
  const Move lastMove = thread->position_.history_.back().move;
  assert(lastMove.from < kNumSquares && lastMove.to < kNumSquares);
  for (ExtMove* move = moves; move < end; ++move) {
    if (move->move == entry.bestMove) {
      move->score = kMaxEval;
      continue;
    }
    assert(move->move.from < kNumSquares);
    assert(move->move.to < kNumSquares);
    assert(move->piece >= Piece::NO_PIECE && move->piece < Piece::NUM_PIECES);
    assert(cp2p(move->capture) < Piece::NUM_PIECES);

    move->score = kQMoveOrderingPieceValue[cp2p(move->capture)];
    move->score -= value_or_zero(
      ((threats.badForOur<TURN>(move->piece) & bb(move->move.to)) > 0),
      kQMoveOrderingPieceValue[move->piece]
    );

    move->score += frame->killers.contains(move->move) ? 8000 : 0;

    move->score += frame->responseTo[move->piece][lastMove.to] == move->move ? 20 : 0;
    move->score += frame->responseFrom[move->piece][lastMove.from] == move->move ? 20 : 0;
    move->score += (frame - 2)->responseTo[move->piece][lastMove.to] == move->move ? 10 : 0;
    move->score += (frame - 2)->responseFrom[move->piece][lastMove.from] == move->move ? 10 : 0;
    move->score += (frame - 4)->responseTo[move->piece][lastMove.to] == move->move ? 5 : 0;
    move->score += (frame - 4)->responseFrom[move->piece][lastMove.from] == move->move ? 5 : 0;
  }
  std::sort(
    moves,
    end,
    [](const ExtMove& a, const ExtMove& b) {
      return a.score > b.score;
    }
  );

  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Ordered moves: ";
    for (ExtMove* m = moves; m != end; ++m) {
      std::cout << m->move.uci() << "(" << m->score << ") ";
    }
    std::cout << std::endl;
  }

  for (ExtMove* move = moves; move < end; ++move) {
    if (move->score <= 0) {
      // Don't consider moves that lose material according to move ordering heuristic.
      continue;
    }
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Trying move: " << move->move.uci() << std::endl;
    }
    make_move<TURN>(&thread->position_, move->move);

    // Move generation can sometimes generate illegal en passant moves.
    if (move->move.moveType == MoveType::EN_PASSANT) {
      const bool inCheck = can_enemy_attack<TURN>(
        thread->position_,
        lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
      );
      if (inCheck) {
        if (IS_PRINT_QNODE) {
          std::cout << repeat("  ", plyFromRoot) << "Illegal en passant move generated: " << move->move.uci() << std::endl;
        }
        undo<TURN>(&thread->position_);
        continue;
      }
    }

    if (can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    )) {
      // Need this check because of en passant captures into check.
      // e.g. b5c6 in position 8/1k6/6R1/KPpr4/8/8/8/8 w - c6 0 62
      if (IS_PRINT_QNODE) {
        std::cout << repeat("  ", plyFromRoot) << "Illegal move generated that leaves us in check: " << move->move.uci() << std::endl;
      }
      undo<TURN>(&thread->position_);
      continue;
    }

    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Calling qsearch recursively." << std::endl;
    }
    auto foo = qsearch<opposite_color<TURN>()>(thread, to_child_eval(beta), to_child_eval(alpha), plyFromRoot + 1, quiescenceDepth + 1, frame + 1, stopThinking).evaluation;
    ColoredEvaluation<TURN> eval = to_parent_eval(foo);
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Eval from recursive qsearch: " << eval.value << "( from " << foo.value << ")" << std::endl;
    }
    undo<TURN>(&thread->position_);
    if (eval > bestResult.evaluation) {
      bestResult.bestMove = move->move;
      bestResult.evaluation = eval;
    }
    if (eval > alpha) {
      alpha = ColoredEvaluation<TURN>(eval.value);
      if (alpha >= beta) {
        assert(move->piece >= Piece::NO_PIECE && move->piece < Piece::NUM_PIECES);
        assert(lastMove.to < 64 && lastMove.from < 64);
        frame->killers.add(move->move);
        frame->responseTo[move->piece][lastMove.to] = move->move;
        frame->responseFrom[move->piece][lastMove.from] = move->move;
        break;
      }
    }
  }

  // Store in Transposition Table
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;

  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Storing in qTT: move=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << " bound=" << bound_type_to_string(bound) << " hash=" << thread->position_.currentState_.hash << " fen=" << thread->position_.fen() << std::endl;
  }
  thread->tt_->store(
    thread->position_.currentState_.hash,
    bestResult.bestMove,
    0,
    bestResult.evaluation.value,
    bound
  );

  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search returning: bestMove=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << " hash=" << thread->position_.currentState_.hash << " fen=" << thread->position_.fen() << std::endl;
  }

  return bestResult;
}

/**
 * Note: if you set stopThinking to true, there is no guarantee that this will return a sensible/valid result.
 * In practice, you will likely want to re-search with depth=1 and stopThinking=false to get a valid move.
 */
template<Color TURN, SearchType SEARCH_TYPE>
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, Frame *frame, std::atomic<bool> *stopThinking) {
  assert(thread->position_.turn_ == TURN);
  const uint64_t key = thread->position_.currentState_.hash;
  frame->hash = key;
  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Negamax called: depth=" << depth << " alpha=" << alpha << " beta=" << beta << " plyFromRoot=" << plyFromRoot << " history " << thread->position_.history_ << std::endl;
  }
  const ColoredEvaluation<TURN> originalAlpha = alpha;

  // Because of how we handle checkmate values, this condition is basically
  // how we avoid looking for mate-in-5 if we already found mate-in-4.
  if (alpha >= beta) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Alpha-beta window is invalid (alpha >= beta). Returning beta." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, beta);
  }

  if (frame - thread->frames_ >= kMaxPlyFromRoot - 1) {
    const auto r = qsearch<TURN>(thread, alpha, beta, plyFromRoot, kMaxQuiescenceDepth, frame, stopThinking);
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Max ply from root reached, returning from quiescence search: " << r << std::endl;
    }
    return r;
  }

  if (depth == 0) {
    auto r = qsearch(thread, alpha, beta, plyFromRoot, 0, frame, stopThinking);
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Returning from quiescence search: " << r << std::endl;
    }
    return r;
  }

  // TODO: Check if any move leads to a draw by repetition.
  // If so, set alpha to kDraw.

  constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
  if (SEARCH_TYPE == SearchType::ROOT) {
    // Normally our parent computes frame->inCheck and passes it down to us,
    // but for the root node we need to compute it ourselves.
    frame->inCheck = can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    );
  }

  if (SEARCH_TYPE != SearchType::ROOT && stopThinking->load()) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Returning (Search stopped externally)." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, originalAlpha);
  }

  // Check if draw by repetition or insufficient material.
  bool isDraw = thread->position_.is_3fold_repetition(plyFromRoot);
  isDraw |= !frame->inCheck && thread->position_.is_fifty_move_rule();
  isDraw |= thread->position_.is_material_draw();
  if (isDraw) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Returning (Draw detected)." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(originalAlpha, beta));
  }

  // Transposition Table probe
  TTEntry entry;
  if (thread->tt_->probe(key, entry)) {
    if (entry.depth >= depth) {
      if (SEARCH_TYPE != SearchType::ROOT) {
        if (entry.bound == BoundType::EXACT) {
          if (IS_PRINT_NODE) {
            std::cout << repeat("  ", plyFromRoot) << "Returning (TT Hit: EXACT; eval=" << entry.bestMove << " " << entry.value << ")" << std::endl;
          }
          return NegamaxResult<TURN>(entry.bestMove, ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta));
        } else if (entry.bound == BoundType::LOWER) {
          if (entry.value >= beta.value) {
            if (IS_PRINT_NODE) {
              std::cout << repeat("  ", plyFromRoot) << "Returning (TT Hit: LOWER)" << std::endl;
            }
            return NegamaxResult<TURN>(entry.bestMove, beta);
          }
          alpha = ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta);
        } else if (entry.bound == BoundType::UPPER) {
          if (entry.value <= alpha.value) {
            if (IS_PRINT_NODE) {
              std::cout << repeat("  ", plyFromRoot) << "Returning (TT Hit: UPPER)" << std::endl;
            }
            return NegamaxResult<TURN>(entry.bestMove, alpha);
          }
          beta = ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta);
        }
      }
    } else {
      // TODO: can we do something with entry.value here?
    }
  } else {
    entry.bestMove = kNullMove;
  }

  thread->nodeCount_++;
  if ((thread->nodeCount_ & 1023) == 0) {
    if (std::chrono::high_resolution_clock::now() >= thread->stopTime_ || thread->nodeCount_ >= thread->nodeLimit_) {
      stopThinking->store(true);
    }
  } else if (thread->nodeCount_ >= thread->nodeLimit_) {
    stopThinking->store(true);
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

  if (moves == end) {
    if (frame->inCheck) {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Returning (Checkmate detected)." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate).clamp_(originalAlpha, beta));
    } else {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Returning (Stalemate detected)." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(Evaluation(kDraw)).clamp_(originalAlpha, beta));
    }
  }

  // Internal Iterative Deepening.
  #ifndef NO_IID
    // If we don't have a best move from the TT, we compute one with reduced depth.
    if (depth > 2 && (entry.key != key || entry.bestMove == kNullMove)) {
      NegamaxResult<TURN> result = negamax<TURN, SEARCH_TYPE>(thread, depth - 2, alpha, beta, plyFromRoot, frame, stopThinking);
      entry.bestMove = result.bestMove;
      entry.value = result.evaluation.value;
    }
  #endif

  // Add score to each move.
  Threats threats;
  create_threats(thread->position_.pieceBitboards_, thread->position_.colorBitboards_, &threats);
  // We never call evaluate in interior nodes, but it behooves us to keep the accumulator
  // up to date so our children/grandchildren can benefit from it.
  frame->staticEval = evaluate<TURN>(thread->position_.evaluator_, thread->position_, threats, plyFromRoot, alpha, beta).value;

  // Razoring.
  //  # PLAYER     :  RATING  ERROR  POINTS  PLAYED   (%)
  //  1 uci-50     :     2.7    1.8  7267.5   14400    50
  //  2 uci-150    :     1.5    2.3  4010.0    8000    50
  //  3 uci-100    :     0.2    1.8  6660.5   13329    50
  //  4 uci-200    :    -1.7    2.9  2388.0    4800    50
  //  5 old        :    -2.6    1.9  5803.0   11729    49
  #if EVAL_AGNOSTIC == 0
  static constexpr int kRazoringMargin = 50;
  if (depth == 1 && frame->staticEval < alpha.value - kRazoringMargin) {
    const auto r = qsearch<TURN>(thread, alpha, beta, plyFromRoot, 0, frame, stopThinking);
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Razoring: static eval is much worse than alpha. Returning from quiescence search: " << r << std::endl;
    }
    // We should check "r.evaluation <= alpha" here, but omitting the check
    // seems to perform better in practice... somehow. Changing the above
    // search to a null-window search also makes us perform worse, which is
    // also quite counterintuitive -- surely a null-window search is pure savings
    // on a move that we're trying to prove bad... right?
    return r;
  }
  // Reverse futility pruning (+29.6 ± 2.7)
  if (depth == 1 && frame->staticEval > beta.value + kRazoringMargin) {
    const auto r = NegamaxResult<TURN>(kNullMove, beta);
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Reverse futility pruning: static eval is much better than beta. Returning beta." << std::endl;
    }
    return r;
  }
  #endif

  // Null move pruning.
  // This is roughly equivalent to having twice as much time.
  //  # PLAYER       :  RATING  ERROR  POINTS  PLAYED   (%)
  //  1 nmp-slow     :    64.1    5.5  2963.5    4800    62
  //  2 nmp-fast     :     2.3    5.4  2420.5    4800    50
  //  3 main-slow    :    -4.1    5.5  2364.0    4800    49
  //  4 main-fast    :   -62.3    5.4  1852.0    4800    39
  const int myPieceCount = std::popcount(thread->position_.colorBitboards_[TURN] & ~thread->position_.pieceBitboards_[coloredPiece<TURN, Piece::PAWN>()]);
  if (SEARCH_TYPE == SearchType::NULL_WINDOW_SEARCH && !frame->inCheck && myPieceCount > 0 && depth > 0) {
    constexpr int reduction = 4;
    const int reducedDepth = std::max(0, depth - reduction);
    make_nullmove<TURN>(&thread->position_);
    ColoredEvaluation<TURN> r = to_parent_eval(negamax<opposite_color<TURN>(), SearchType::NULL_WINDOW_SEARCH>(
      thread, reducedDepth, to_child_eval(beta), to_child_eval(beta - 1), plyFromRoot + 1, frame + 1, stopThinking
    ).evaluation);
    undo_nullmove<TURN>(&thread->position_);
    if (r >= beta) {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Null move pruning: null move search returned " << r.value << " which is >= beta. Returning beta." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, beta);
    }
  }

  const Move lastMove = SEARCH_TYPE == SearchType::ROOT ? kNullMove : thread->position_.history_.back().move;
  // Move ordering operates in bands
  // +8000: is capture
  // +8000: is killer move
  // deltas for ranking captures can range from -4000 to 4000
  // This way, the killer move is tried after all the non-sacking captures, but before any of the bad captures. See https://www.chessprogramming.org/Move_Ordering#Captures for more discussion.
  static constexpr Evaluation kMoveOrderingPieceValue[Piece::NUM_PIECES] = {
    0,    // NO_PIECE
    100,  // PAWN
    320,  // KNIGHT
    330,  // BISHOP
    500,  // ROOK
    900,  // QUEEN
    2000 // KING
  };

  const SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[coloredPiece<opposite_color<TURN>(), Piece::KING>()]);
  const CheckMap checkMap = compute_potential_attackers<TURN>(thread->position_, theirKingSq);

  for (ExtMove* move = moves; move < end; ++move) {
    if (move->move == entry.bestMove) {
      move->score = kMaxEval;
      continue;
    }
    move->score = 0;

    // Prioritize captures after the TT move.
    move->score += move->capture != ColoredPiece::NO_COLORED_PIECE ? 8000 : 0;

    // Bonus for moves that give check.
    move->score += checkMap.data[move->piece] & bb(move->move.to) ? 100 : 0;

    // Ranking within captures. Bonus for capturing a high value piece, penalty for
    // taking a piece that is defended.
    move->score += kMoveOrderingPieceValue[cp2p(thread->position_.tiles_[move->move.to])];
    move->score -= value_or_zero(
      ((threats.badForOur<TURN>(move->piece) & bb(move->move.to)) > 0)
      &&
      move->capture != ColoredPiece::NO_COLORED_PIECE
    , kMoveOrderingPieceValue[move->piece]);

    // Prioritize the killer move(s) as equivalent to a non-sacking capture.
    move->score += frame->killers.contains(move->move) ? 8000 : 0;

    if (move->capture == ColoredPiece::NO_COLORED_PIECE) {
      move->score += thread->quietHistory_[move->piece][move->move.to] / 64;
    } else {
      move->score += thread->captureHistory_[move->piece][cp2p(thread->position_.tiles_[move->move.to])][move->move.to] / 64;
    }

    // Penalize non-capture moves that move to a defended square.
    move->score -= value_or_zero(
      ((threats.badForOur<TURN>(move->piece) & bb(move->move.to)) > 0)
      &&
      move->capture == ColoredPiece::NO_COLORED_PIECE
    , 200);
    // Bonus for moving a piece that is under attack.
    move->score += value_or_zero(
      ((threats.badForOur<TURN>(move->piece) & bb(move->move.from)) > 0)
    , 50);

    // Prioritize moves that caused a beta cutoff in a similar position, in response to a similar move.
    move->score += frame->responseTo[move->piece][lastMove.to] == move->move ? 20 : 0;
    move->score += frame->responseFrom[move->piece][lastMove.from] == move->move ? 20 : 0;
    move->score += (frame - 2)->responseTo[move->piece][lastMove.to] == move->move ? 10 : 0;
    move->score += (frame - 2)->responseFrom[move->piece][lastMove.from] == move->move ? 10 : 0;
    move->score += (frame - 4)->responseTo[move->piece][lastMove.to] == move->move ? 5 : 0;
    move->score += (frame - 4)->responseFrom[move->piece][lastMove.from] == move->move ? 5 : 0;

    // Penalize pawn moves.
    move->score -= move->piece == Piece::PAWN;
  }
  std::sort(
    moves,
    end,
    [](const ExtMove& a, const ExtMove& b) {
      return a.score > b.score;
    }
  );

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Ordered moves: ";
    for (ExtMove* m = moves; m != end; ++m) {
      std::cout << m->move.uci() << "(" << m->score << ") ";
    }
    std::cout << std::endl;
  }

  if (SEARCH_TYPE == SearchType::ROOT) {
    thread->primaryVariations_.clear();
  }

  const Bitboard theirTargets = (TURN == Color::WHITE) ? threats.blackTargets : threats.whiteTargets;
  const Bitboard theirPawns = (TURN == Color::WHITE) ? 
    thread->position_.pieceBitboards_[coloredPiece<Color::BLACK, Piece::PAWN>()] : 
    thread->position_.pieceBitboards_[coloredPiece<Color::WHITE, Piece::PAWN>()];
  const Bitboard aheadOfTheirPawns = ((TURN == Color::WHITE) ? 
    southFill(theirPawns) : northFill(theirPawns)) & ~theirPawns;
  const Bitboard ourPassedPawnMask = ~(fatten(aheadOfTheirPawns));

  // We use kMinEval instead of alpha so that we still get a best move, even if all moves fail low.
  // This is helpful for probing the TT to try and understand why we got a cutoff. I don't think it
  // meaningfully changes the engine's strength, since if all moves fail low, then the TT will just
  // store the first move. OTOH maybe this is bad, since prioritizing a random first move could be
  // worse than allowing the other move ordering heuristics to just do their thing.
  //
  // Note that, regardless of whether we initialize bestResult.evaluation with kMinEval or alpha,
  // it is different than alpha! Alpha is the value passed to our children (as -beta), whereas
  // bestResult.evaluation is the value we will return. When multiPV > 1 and we're in the root,
  // these are typically different values.
  NegamaxResult<TURN> bestResult(kNullMove, ColoredEvaluation<TURN>(kMinEval));
  for (ExtMove* move = moves; move < end; ++move) {
    constexpr ColoredPiece enemyKing = coloredPiece<opposite_color<TURN>(), Piece::KING>();
    assert((thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) == 0);
    make_move<TURN>(&thread->position_, move->move);

    // TODO: pass this to child frame.
    const bool areWeInCheck = can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    );
    if (areWeInCheck) {
      // Need this check because of en passant captures into check.
      // e.g. b5c6 in position 8/1k6/6R1/KPpr4/8/8/8/8 w - c6 0 62
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Illegal move generated that leaves us in check: " << move->move.uci() << std::endl;
      }
      undo<TURN>(&thread->position_);
      continue;
    }
    const bool moveGivesCheck = can_enemy_attack<opposite_color<TURN>()>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[enemyKing])
    );
    (frame + 1)->inCheck = moveGivesCheck;

    ColoredEvaluation<TURN> eval;
    int childDepth = depth - 1;

    // Don't reduce depth for sensible captures (Elo difference: 254.7 +/- 286.2, LOS: 98.7 %)
    bool isGoodCapture = move->capture != ColoredPiece::NO_COLORED_PIECE && cp2p(move->capture) > move->piece;
    // Also don't reduce depth for safe passed pawn pushes.
    bool isSafePassedPawnPush = move->piece == Piece::PAWN && (ourPassedPawnMask & ~theirTargets & bb(move->move.to)) > 0;
    // if (isGoodCapture || isSafePassedPawnPush) {
    //   childDepth += 1;
    // }

    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Recursing with childDepth=" << childDepth << " (hash=" << thread->position_.currentState_.hash << "; move=" << move->move.uci() << "; alpha=" << alpha.value << "; beta=" << beta.value << ")" << std::endl;
    }

    const int index = move - moves;

    const bool isQuiet = (
      (move->capture == ColoredPiece::NO_COLORED_PIECE) && (move->move.moveType != MoveType::PROMOTION)
    );

    // TODO: we can probably remove the "not checkmating" check here, but we need to be careful since null window bounds,
    // as they are currently written, can be equal! If you want to remove the "not checkmating" condition, you should test
    // with
    // $ ./uci "position fen r5k1/3Q1p2/2p3pp/4b3/p7/P1P1q3/1rBR2bP/1K1R4 w - - 0 26 moves b1a1 e3c3" "go depth 4" "lazyquit"
    if (move->move != moves[0].move && (SEARCH_TYPE != SearchType::ROOT || thread->multiPV_ == 1) && alpha.value > kLongestForcedMate && alpha.value < -kLongestForcedMate) {
      #ifndef NO_LMR
        int lateMoveReduction = childDepth >= 3 && index >= 3;
        lateMoveReduction += index > 8 ? 1 : 0;
        lateMoveReduction -= isGoodCapture ? 1 : 0;
        lateMoveReduction -= isSafePassedPawnPush ? 1 : 0;
        const int reducedChildDepth = std::max(childDepth - std::max(0, lateMoveReduction), 0);
      #else
        const int reducedChildDepth = childDepth;
      #endif

      eval = to_parent_eval(negamax<opposite_color<TURN>(), SearchType::NULL_WINDOW_SEARCH>(thread, reducedChildDepth, to_child_eval(alpha + 1), to_child_eval(alpha), plyFromRoot + 1, frame + 1, stopThinking).evaluation);
      if (eval.value > alpha.value) {
        if (IS_PRINT_NODE) {
          std::cout << repeat("  ", plyFromRoot) << "Null window search failed; doing full window search." << std::endl;
        }
        constexpr SearchType searchType = SEARCH_TYPE == SearchType::ROOT ? SearchType::NORMAL_SEARCH : SEARCH_TYPE;
        eval = to_parent_eval(negamax<opposite_color<TURN>(), searchType>(thread, childDepth, to_child_eval(beta), to_child_eval(alpha), plyFromRoot + 1, frame + 1, stopThinking).evaluation);
      }
    } else {
      // Simple, full-window, full-depth search. Used for the first move in non-root search.
      // In the root node, we use this when multiPV==1, since we don't care about the exact
      // evaluation of moves that aren't the best move.
      constexpr SearchType firstMoveSearchType = SEARCH_TYPE == SearchType::ROOT ? SearchType::NORMAL_SEARCH : SEARCH_TYPE;
      eval = to_parent_eval(negamax<opposite_color<TURN>(), firstMoveSearchType>(thread, childDepth, to_child_eval(beta), to_child_eval(alpha), plyFromRoot + 1, frame + 1, stopThinking).evaluation);
    }

    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Move " << move->move.uci() << " has evaluation ";
      if (eval <= alpha) {
        std::cout << "≤" << alpha.value;
      } else if (eval >= beta) {
        std::cout << "≥" << beta.value;
      } else {
        std::cout << "=" << eval.value;
      }
      std::cout << " (" << alpha.value << " ≤ " << eval.value << " ≤ " << beta.value << ") " << thread->position_.history_ << std::endl;
    }

    undo<TURN>(&thread->position_);
    if (eval > bestResult.evaluation) {
      bestResult.bestMove = move->move;
      bestResult.evaluation = eval;
    }
    if (eval > alpha) {
      if (SEARCH_TYPE == SearchType::ROOT) {
        // In multi-PV search, we want to keep track of multiple best moves and
        // only raise alpha if we have the top N moves.

        // We don't really care about optimizing this too much since it only happens
        // at the root of the search.
        thread->primaryVariations_.push_back(std::make_pair(move->move, eval.value));
        std::stable_sort(
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
        // TODO: check if this move is quiet. Probably also check if we've already added it as a killer.
        if (isQuiet) {
          int32_t bonus = std::min(depth * depth, 400);
          int32_t& hist = thread->quietHistory_[move->piece][move->move.to];
          hist += bonus - hist * std::abs(bonus) / 16384;
        } else {
          int32_t bonus = std::min(depth * depth, 400);
          int32_t& hist = thread->captureHistory_[move->piece][cp2p(move->capture)][move->move.to];
          hist += bonus - hist * std::abs(bonus) / 16384;
        }
        frame->killers.add(move->move);
        frame->responseTo[move->piece][lastMove.to] = move->move;
        frame->responseFrom[move->piece][lastMove.from] = move->move;
        break;
      }
    }
  }

  if (stopThinking->load()) {
    // Search was stopped externally. We cannot trust the result
    // of our for loop above, so we return early to avoid writing
    // an inaccurate result to the transposition table.
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Returning (Search stopped externally)." << std::endl;
    }
    return bestResult;
  }

  // Store in Transposition Table
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;

  bestResult.evaluation.clamp_(originalAlpha, beta);

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Storing in TT: depth=" << depth << " move=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << " bound=" << bound_type_to_string(bound) << " hash=" << thread->position_.currentState_.hash << " fen=" << thread->position_.fen() << std::endl;
  }
  thread->tt_->store(
    thread->position_.currentState_.hash,
    bestResult.bestMove,
    depth,
    bestResult.evaluation.value,
    bound
  );

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Negamax returning: bestMove=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value  << " depth=" << depth << std::endl;
  }

  return bestResult;
}

}  // namespace ChessEngine

#endif  // NEGAMAX_H
