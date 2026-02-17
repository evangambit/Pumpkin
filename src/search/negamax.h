/**
 * This file contains the core search logit, but may not be the most
 * usable API. See search.h for a more user-friendly search interface.
 */

#ifndef NEGAMAX_H
#define NEGAMAX_H

#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <memory>
#include <unordered_set>

#include "../game/Position.h"
#include "../game/Utils.h"
#include "../game/movegen/movegen.h"
#include "../game/Threats.h"
#include "../eval/Evaluator.h"
#include "../eval/ColoredEvaluation.h"

#include "transposition_table.h"


// IS_PRINT_NODE and IS_PRINT_QNODE can be used to print debugging information for specific nodes.

#ifndef IS_PRINT_NODE
#define IS_PRINT_NODE 0
// #define IS_PRINT_NODE (SEARCH_TYPE == SearchType::ROOT)
// #define IS_PRINT_NODE (plyFromRoot == 0 || frame->hash == 15932567610229845462ULL || frame->hash == 15427882709703266013ULL)
// #define IS_PRINT_NODE (thread->position_.currentState_.hash == 412260009870427727ULL)
#endif

#ifndef IS_PRINT_QNODE
#define IS_PRINT_QNODE 0
// #define IS_PRINT_QNODE ((frame - quiescenceDepth)->hash == 17514877330620511575ULL)
#endif

namespace ChessEngine {

constexpr unsigned kMaxSearchDepth = 32;
constexpr unsigned kMaxPlyFromRoot = kMaxSearchDepth + 16;
constexpr int kMaxQuiescenceDepth = kMaxPlyFromRoot - kMaxSearchDepth;

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
    } else {
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

  // This pointer should be considered non-owning. The TranspositionTable should created and
  // managed elsewhere since it should be shared across threads and searches.
  TranspositionTable* tt_;

  Thread(
    uint64_t id,
    const Position& pos,
    uint64_t multiPV,
    const std::unordered_set<Move>& permittedMoves,
    TranspositionTable* tt
  ) : id_(id), position_(pos), permittedMoves_(permittedMoves), multiPV_(multiPV), tt_(tt) {}
  
  Thread(const Thread& other)
  : id_(other.id_),
    multiPV_(other.multiPV_),
    depth_(other.depth_),
    position_(other.position_),
    permittedMoves_(other.permittedMoves_),
    primaryVariations_(other.primaryVariations_),
    nodeCount_(other.nodeCount_),
    qNodeCount_(other.qNodeCount_),
    nodeLimit_(other.nodeLimit_),
    tt_(other.tt_) {}

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

template<Color TURN>
NegamaxResult<TURN> qsearch(Thread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, int quiescenceDepth, Frame *frame, std::atomic<bool> *stopThinking) {
  // Prevent stack overflow - return static eval if we've gone too deep
  if (quiescenceDepth >= kMaxQuiescenceDepth) {
    return NegamaxResult<TURN>(kNullMove, evaluate<TURN>(thread->position_.evaluator_, thread->position_));
  }

  frame->hash = thread->position_.currentState_.hash;
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search called: alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " quiescenceDepth=" << quiescenceDepth << " history" << thread->position_.history_ << std::endl;
  }

  // Transposition Table probe
  TTEntry entry;
  uint64_t key = thread->position_.currentState_.hash;
  if (thread->tt_->probe(key, entry)) {
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
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate).clamp_(originalAlpha, beta));
  }
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Comparing static evaluation to alpha/beta" << std::endl;
  }

  NegamaxResult<TURN> bestResult(kNullMove, evaluate<TURN>(thread->position_.evaluator_, thread->position_).clamp_(alpha, beta));
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Static evaluation: " << bestResult.evaluation.value << " (hash = " << frame->hash << ")" << std::endl;
  }
  if (!inCheck) {
    if (bestResult.evaluation >= beta) {
      return bestResult;
    }
    if (bestResult.evaluation > alpha) {
      alpha = bestResult.evaluation;
    }
  }

  // Move ordering: captures that capture higher value pieces first.
  assert(!thread->position_.history_.empty() && "qsearch requires history to have at least one move");
  const Move lastMove = thread->position_.history_.back().move;
  assert(lastMove.from < 64 && lastMove.to < 64);
  Threats<TURN> threats(thread->position_);
  for (ExtMove* move = moves; move < end; ++move) {
    if (move->move == entry.bestMove) {
      move->score = kMaxEval;
      continue;
    }
    assert(move->move.from < 64);
    assert(move->move.to < 64);
    assert(move->piece >= Piece::NO_PIECE && move->piece < Piece::NUM_PIECES);
    Piece capturedPiece = cp2p(thread->position_.tiles_[move->move.to]);
    assert(capturedPiece < Piece::NUM_PIECES);
    assert(cp2p(move->capture) < Piece::NUM_PIECES);

    move->score = kQMoveOrderingPieceValue[cp2p(move->capture)];
    move->score -= value_or_zero(
      ((threats.badForOur[move->piece] & bb(move->move.to)) > 0),
      kQMoveOrderingPieceValue[move->piece]
    );

    move->score += thread->frames_[plyFromRoot].killers.contains(move->move) ? 8000 : 0;

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
    if (move->score < 0) {
      // Don't consider moves that lose material according to move ordering heuristic.
      continue;
    }
    if (thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) {
      undo<TURN>(&thread->position_);
      std::cout << "Illegal move generated in qsearch: " << move->move.uci() << std::endl;
      std::cout << thread->position_.fen() << std::endl;
      std::cout << thread->position_ << std::endl;
      continue;
    }
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Trying move: " << move->move.uci() << std::endl;
    }
    make_move<TURN>(&thread->position_, move->move);
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Position after move: " << thread->position_.fen() << std::endl;
    }

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

    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Calling qsearch recursively." << std::endl;
    }
    ColoredEvaluation<TURN> eval = -qsearch<opposite_color<TURN>()>(thread, -beta, -alpha, plyFromRoot + 1, quiescenceDepth + 1, frame + 1, stopThinking).evaluation;
    if (eval.value < kLongestForcedMate) {
      eval.value += 1;
    } else if (eval.value > -kLongestForcedMate) {
      eval.value -= 1;
    }
    if (IS_PRINT_QNODE) {
      std::cout << repeat("  ", plyFromRoot) << "Eval from recursive qsearch: " << eval.value << std::endl;
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

  thread->tt_->store(
    thread->position_.currentState_.hash,
    bestResult.bestMove,
    0,
    bestResult.evaluation.value,
    bound
  );

  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search returning: bestMove=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << std::endl;
  }

  return bestResult;
}

template<Color TURN, SearchType SEARCH_TYPE>
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, Frame *frame, std::atomic<bool> *stopThinking) {
  assert(thread->position_.turn_ == TURN);
  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Negamax called: depth=" << depth << " alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " history " << thread->position_.history_ << std::endl;
  }
  const ColoredEvaluation<TURN> originalAlpha = alpha;
  const uint64_t key = thread->position_.currentState_.hash;
  frame->hash = key;

  // Check for immediate cutoffs based on mate distance.
  Evaluation lowestPossibleEvaluation = kCheckmate + plyFromRoot;
  if (SEARCH_TYPE != SearchType::ROOT && lowestPossibleEvaluation >= beta.value) {
    return NegamaxResult<TURN>(kNullMove, beta);
  }
  Evaluation highestPossibleEvaluation = -kCheckmate - plyFromRoot;
  if (SEARCH_TYPE != SearchType::ROOT && highestPossibleEvaluation <= alpha.value) {
    return NegamaxResult<TURN>(kNullMove, alpha);
  }

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Probing TT with key " << thread->position_.currentState_.hash << std::endl;
  }

  // Transposition Table probe
  TTEntry entry;
  if (thread->tt_->probe(key, entry) && entry.depth >= depth) {
    if (SEARCH_TYPE != SearchType::ROOT) {
      if (entry.bound == BoundType::EXACT) {
        if (IS_PRINT_NODE) {
          std::cout << repeat("  ", plyFromRoot) << "TT Hit: EXACT" << std::endl;
        }
        return NegamaxResult<TURN>(entry.bestMove, ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta));
      } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
        if (IS_PRINT_NODE) {
          std::cout << repeat("  ", plyFromRoot) << "TT Hit: LOWER" << std::endl;
        }
        return NegamaxResult<TURN>(entry.bestMove, beta);
      } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
        if (IS_PRINT_NODE) {
          std::cout << repeat("  ", plyFromRoot) << "TT Hit: UPPER" << std::endl;
        }
        return NegamaxResult<TURN>(entry.bestMove, alpha);
      }
    }
  } else {
    entry.bestMove = kNullMove;
  }

  if (SEARCH_TYPE != SearchType::ROOT && stopThinking->load()) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Search stopped externally." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(0).clamp_(originalAlpha, beta));
  }

  // TODO: We need to check this *after* we do the checkmate test above, since you can win on the 50th move.
  if (thread->position_.is_fifty_move_rule()) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Fifty-move rule draw detected." << std::endl;
    }
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(std::max(originalAlpha.value, std::min(beta.value, Evaluation(0)))));
  }

  thread->nodeCount_++;
  if (thread->nodeCount_ >= thread->nodeLimit_) {
    stopThinking->store(true);
  }

  if (depth == 0) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Entering quiescence search." << std::endl;
    }
    return qsearch(thread, alpha, beta, plyFromRoot, 0, frame, stopThinking);
  }

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Generating moves." << std::endl;
  }

  ExtMove moves[kMaxNumMoves];
  ExtMove* end;
  if (SEARCH_TYPE == SearchType::ROOT) {
    end = compute_legal_moves<TURN>(&thread->position_, moves);
  } else {
    end = compute_moves<TURN, MoveGenType::ALL_MOVES>(thread->position_, moves);
  }


  if (SEARCH_TYPE == SearchType::ROOT) {
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Filtering moves" << std::endl;
    }
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

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Checking for checkmate/stalemate/fifty-move rule." << std::endl;
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
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate));
    } else {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Stalemate detected." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(std::max(originalAlpha.value, std::min(beta.value, Evaluation(0)))));
    }
  }

  // Internal Iterative Deepening.
  // If we don't have a best move from the TT, we compute one with reduced depth.
  if (depth > 2 && (entry.key != key || entry.bestMove == kNullMove)) {
    NegamaxResult<TURN> result = negamax<TURN, SEARCH_TYPE>(thread, depth - 2, alpha, beta, plyFromRoot, frame, stopThinking);
    entry.bestMove = result.bestMove;
    entry.value = result.evaluation.value;
  }

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Ordering moves." << std::endl;
  }

  // Add score to each move.
  Threats<TURN> threats(thread->position_);
  const Move lastMove = SEARCH_TYPE == SearchType::ROOT ? kNullMove : thread->position_.history_.back().move;
  /**
  ±16000: best move from transposition table
  +8000: is capture
  
  deltas for ranking captures can range from -4000 to 4000

   */
  constexpr Evaluation kMoveOrderingPieceValue[Piece::NUM_PIECES] = {
    0,    // NO_PIECE
    100,  // PAWN
    320,  // KNIGHT
    330,  // BISHOP
    500,  // ROOK
    900,  // QUEEN
    2000 // KING
  };

  for (ExtMove* move = moves; move < end; ++move) {
    move->score = move->move == entry.bestMove ? 16000 : -16000;
    // Prioritize captures after the TT move.
    move->score += move->capture != ColoredPiece::NO_COLORED_PIECE ? 8000 : -8000;

    // Ranking within captures. Bonus for capturing a high value piece, penalty for
    // taking a piece that is defended.
    move->score += kMoveOrderingPieceValue[cp2p(thread->position_.tiles_[move->move.to])];
    move->score -= value_or_zero(
      ((threats.badForOur[move->piece] & bb(move->move.to)) > 0)
      &&
      move->capture != ColoredPiece::NO_COLORED_PIECE
    , kMoveOrderingPieceValue[move->piece]);

    // Prioritize the killer move(s) as equivalent to a non-sacking capture.
    move->score += thread->frames_[plyFromRoot].killers.contains(move->move) ? 8000 : 0;

    // Penalize non-capture moves that move to a defended square.
    move->score -= value_or_zero(
      ((threats.badForOur[move->piece] & bb(move->move.to)) > 0)
      &&
      move->capture == ColoredPiece::NO_COLORED_PIECE
    , 200);
    // Bonus for moving a piece that is under attack.
    move->score += value_or_zero(
      ((threats.badForOur[move->piece] & bb(move->move.from)) > 0)
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
  }

  if (SEARCH_TYPE == SearchType::ROOT) {
    thread->primaryVariations_.clear();
  }

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Searching moves." << std::endl;
  }

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
  uint8_t numQuietMovesSearched = 0;
  for (ExtMove* move = moves; move < end; ++move) {
    if (thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) {
      // Don't capture the king. TODO: remove this check by fixing move generation.
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Skipping illegal move " << move->move.uci() << " that captures the king: " << move->move.uci() << std::endl;
      }
      continue;
    }
    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Making move " << move->move.uci() << std::endl;
      make_move<TURN>(&thread->position_, move->move);
      std::cout << repeat("  ", plyFromRoot) << "hash after move: " << thread->position_.currentState_.hash << std::endl;
      std::cout << repeat("  ", plyFromRoot) << "history: " << thread->position_.history_ << std::endl;
    } else {
      make_move<TURN>(&thread->position_, move->move);
    }

    constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    const bool inCheck = can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    );
    if (inCheck) {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Illegal move generated that leaves us in check: " << move->move.uci() << std::endl;
      }
      undo<TURN>(&thread->position_);
      std::cout << "Illegal move generated in negamax search: " << move->move.uci() << std::endl;
      std::cout << thread->position_.fen() << std::endl;
      std::cout << thread->position_ << std::endl;
      continue;
    }

    ColoredEvaluation<TURN> eval(0);
    int childDepth = depth - 1;

    // Don't reduce depth for sensible captures (Elo difference: 254.7 +/- 286.2, LOS: 98.7 %)
    if (move->capture != ColoredPiece::NO_COLORED_PIECE && cp2p(move->capture) > move->piece) {
      childDepth += 1;
    }

    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Recursing with childDepth=" << childDepth << " (hash=" << thread->position_.currentState_.hash << ")" << std::endl;
    }

    if (move->move != moves[0].move && (SEARCH_TYPE != SearchType::ROOT || thread->multiPV_ == 1)) {
      // Very conservative late move reduction (115.2 +/- 103.0, LOS: 99.0 %)
      const int reduction = SEARCH_TYPE != SearchType::ROOT
        && childDepth >= 3
        && move->capture == ColoredPiece::NO_COLORED_PIECE
        && move->piece != Piece::PAWN
        && numQuietMovesSearched >= 10
        && !inCheck;

      // Null window search (possibly with reduced depth).
      if (SEARCH_TYPE != SearchType::NULL_WINDOW_SEARCH) {
        eval = -negamax<opposite_color<TURN>(), SearchType::NULL_WINDOW_SEARCH>(thread, std::max(childDepth - reduction, 0), -(alpha + 1), -alpha, plyFromRoot + 1, frame + 1, stopThinking).evaluation;
        if (eval.value > alpha.value) {
          eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, childDepth, -beta, -alpha, plyFromRoot + 1, frame + 1, stopThinking).evaluation;
        }
      } else {
        eval = -negamax<opposite_color<TURN>(), SearchType::NULL_WINDOW_SEARCH>(thread, childDepth, -beta, -alpha, plyFromRoot + 1, frame + 1, stopThinking).evaluation;
      }
    } else {
      // Simple, full-window, full-depth search. Used for the first move in non-root search.
      // In the root node, we use this when multiPV==1, since we don't care about the exact
      // evaluation of moves that aren't the best move.
      eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, childDepth, -beta, -alpha, plyFromRoot + 1, frame + 1, stopThinking).evaluation;
    }

    if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Move " << move->move.uci() << " has evaluation " << eval.value << std::endl;
    }

    // We adjust mate scores to reflect the distance to mate here, rather than when we return kCheckmate. The
    // reason is that otherwise our transposition table scores will reflect the conditions in which they happened
    // to be computed, which wouldn't be accurate if they're being probed further from the root. An alternative
    // solution is to adjust scores before storing them in the transposition table (and unadjusting after probing
    // /them), but this seems simpler.
    if (eval.value < kLongestForcedMate) {
      eval.value += 1;
    } else if (eval.value > -kLongestForcedMate) {
      eval.value -= 1;
    }

    undo<TURN>(&thread->position_);
    numQuietMovesSearched += (move->capture == ColoredPiece::NO_COLORED_PIECE) && (move->move.moveType != MoveType::PROMOTION);
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
        // TODO: check if this move is quiet. Probably also check if we've already added it as a killer.
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
    if (thread->tt_->probe(key, entry)) {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Search stopped externally. Returning TT best move." << std::endl;
      }
      return NegamaxResult<TURN>(entry.bestMove, ColoredEvaluation<TURN>(entry.value).clamp_(originalAlpha, beta));
    } else {
      if (IS_PRINT_NODE) {
        std::cout << repeat("  ", plyFromRoot) << "Search stopped externally. No TT entry found, returning null move." << std::endl;
      }
      if (SEARCH_TYPE == SearchType::ROOT) {
        // Need to always return something sensible from the root.
        // TODO: this is pretty hacky, since nodeCount is checked inside of negamax, but other conditions
        // are checked outside of it. We should probably unify this logic better.
        size_t nodeCount = thread->nodeCount_;
        size_t nodeLimit = thread->nodeLimit_;
        thread->nodeCount_ = 0;
        thread->nodeLimit_ = 10'000'000; // Arbitrary large number to ensure we search at least a little bit.
        std::atomic<bool> neverStop{false};
        NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(thread, 1, ColoredEvaluation<TURN>(kMinEval), ColoredEvaluation<TURN>(kMaxEval), plyFromRoot, frame, &neverStop);
        thread->nodeCount_ += nodeCount;
        thread->nodeLimit_ = nodeLimit;
        return result;
      }
      return NegamaxResult<TURN>(kNullMove, originalAlpha);
    }
  }

  // Store in Transposition Table
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;

  bestResult.evaluation.clamp_(originalAlpha, beta);

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Storing in TT: depth=" << depth << " move=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value << " bound=" << bound_type_to_string(bound) << std::endl;
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