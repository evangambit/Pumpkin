#ifndef SEARCH_H
#define SEARCH_H

#ifndef IS_PRINT_NODE
#define IS_PRINT_NODE 0
// #define IS_PRINT_NODE (thread->position_.currentState_.hash == 412260009870427727ULL)
#endif

#ifndef IS_PRINT_QNODE
#define IS_PRINT_QNODE 0
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <functional>
#include <memory>
#include <unordered_set>

#include "../game/Position.h"
#include "../game/utils.h"
#include "../game/movegen/movegen.h"
#include "../game/Threats.h"
#include "../eval/evaluator.h"
#include "../eval/ColoredEvaluation.h"

#include "transposition_table.h"

namespace ChessEngine {

const int16_t kMoveOrderFromSquare[64 * 6] = {
  // pawn
  109,  109,  106,  107,  106,  103,  108,  106,
  103,  109,  109,  106,  101,  106,  107,  105,
  106,  105,  113,  104,  105,  109,  104,  107,
  108,  108,  106,  108,  107,  106,  102,  109,
   68,   88,   84,   94,   92,   87,   87,   59,
   68,   65,   96,   91,   90,  107,   73,   66,
  -17,  -17,   12,  108,  104,    5,  -10,  -13,
  105,  108,  108,  107,  111,  108,  111,  109,
  // knight
   21,   15,    5,   -1,   -2,    8,   16,   25,
    1,   -4,  -14,  -10,  -15,  -10,  -10,    8,
   -8,  -18,  -18,  -20,  -19,  -18,  -23,   -5,
  -21,  -25,  -24,  -27,  -26,  -20,  -22,  -22,
  -29,  -22,  -25,  -24,  -27,  -29,  -23,  -26,
  -23,  -18,  -26,  -25,  -24,  -23,  -17,  -26,
  -17,  -19,  -20,  -23,  -29,  -23,  -15,  -16,
   18,  -24,  -17,  -19,  -21,  -17,  -25,    9,
  // bishop
    2,  -10,  -11,  -13,   -8,  -15,  -11,   -2,
   -6,  -17,  -20,  -21,  -16,  -14,  -11,  -13,
  -12,  -25,  -26,  -23,  -24,  -29,  -22,  -17,
  -14,  -25,  -20,  -24,  -21,  -22,  -26,  -14,
  -22,  -26,  -23,  -23,  -28,  -25,  -25,  -18,
  -31,  -24,  -18,  -25,  -26,  -26,  -20,  -21,
  -15,  -22,  -28,  -29,  -27,  -23,  -26,  -17,
  -14,  -19,  -27,  -20,  -18,  -28,  -14,  -15,
  // rook
  -27,  -27,  -17,  -26,  -24,  -20,  -25,  -27,
  -25,  -19,  -23,  -22,  -26,  -20,  -14,  -25,
  -23,  -24,  -24,  -19,  -21,  -17,  -22,  -21,
  -22,  -20,  -30,  -21,  -25,  -27,  -32,  -21,
  -26,  -19,  -23,  -22,  -28,  -21,  -22,  -29,
  -22,  -26,  -29,  -18,  -24,  -19,  -30,  -29,
  -22,  -23,  -27,  -18,  -27,  -24,  -22,  -25,
  -27,  -28,  -25,  -22,  -21,  -21,  -26,  -25,
  // queen
   -4,   -2,   -8,   -8,   -8,   -1,   -3,   -4,
   -9,  -10,   -9,   -7,   -7,  -10,  -11,   -6,
  -16,  -17,  -13,  -19,  -17,  -15,  -12,  -15,
  -18,  -13,  -14,  -15,  -22,  -18,  -16,  -19,
  -25,  -17,  -25,  -22,  -30,  -24,  -24,  -27,
  -19,  -26,  -13,  -27,  -33,  -23,  -23,  -22,
  -18,  -25,  -24,  -26,  -27,  -30,  -26,  -12,
  -16,  -16,  -28,  -29,  -32,  -21,  -21,  -12,
  // king
   19,   10,    8,    8,   10,    7,    7,   25,
    4,  -12,   -9,   -8,  -11,  -13,    0,    7,
   -8,  -16,  -16,  -17,  -11,  -17,  -15,    0,
  -11,  -23,  -17,  -19,  -21,  -19,  -23,   -5,
  -21,  -24,  -25,  -21,  -21,  -21,  -22,  -22,
  -23,  -16,  -26,  -21,  -25,  -24,  -28,  -18,
  -26,  -25,  -25,  -27,  -24,  -25,  -27,  -28,
  -23,  -27,  -30,  -25,  -23,  -18,  -20,  -21,
};
const int16_t kMoveOrderToSquare[64 * 6] = {
// pawn
 105,  104,  110,  111,  109,  109,  113,  106,
 108,  112,  103,  113,  107,  114,  114,  106,
 109,  112,  115,  114,  117,  112,  107,  119,
 103,  112,  108,  111,  108,  107,  105,  100,
  34,   31,   77,  102,  101,   72,   24,   27,
  71,   80,  111,  110,  108,  115,   83,   68,
 109,  107,  110,  110,  113,  113,  106,  104,
 111,  108,  111,  110,  108,  109,  111,  110,
// knight
 -10,  -13,  -15,  -13,  -17,  -20,  -16,   -4,
 -17,  -21,  -23,  -19,  -17,  -25,  -12,  -19,
 -16,  -14,  -25,  -20,  -23,  -18,  -15,  -22,
 -13,  -26,  -19,  -22,  -31,  -22,  -21,  -17,
 -20,  -17,  -21,  -23,  -26,  -23,  -20,  -25,
 -19,  -22,  -25,  -23,  -25,  -18,  -21,  -21,
 -23,  -14,  -20,  -22,  -26,  -23,  -23,  -14,
 -19,  -28,  -23,  -17,  -27,  -24,  -21,  -25,
// bishop
 -15,  -19,  -18,  -23,  -27,  -17,  -23,  -19,
 -20,  -22,  -14,  -21,  -24,  -22,  -18,  -25,
 -17,  -20,  -23,  -20,  -21,  -20,  -25,  -21,
 -13,  -22,  -22,  -19,  -26,  -23,  -21,  -22,
 -25,  -20,  -26,  -16,  -21,  -26,  -18,  -20,
 -19,  -23,  -20,  -21,  -24,  -22,  -29,  -27,
 -21,  -22,  -22,  -17,  -22,  -21,  -19,  -24,
 -21,  -19,  -21,  -26,  -21,  -22,  -25,  -21,
// rook
 -25,  -20,  -28,  -24,  -24,  -20,  -20,  -23,
 -20,  -26,  -20,  -20,  -20,  -18,  -24,  -29,
 -24,  -24,  -18,  -15,  -19,  -24,  -20,  -24,
 -21,  -27,  -23,  -25,  -18,  -16,  -24,  -17,
 -19,  -25,  -17,  -18,  -18,  -16,  -27,  -21,
 -14,  -23,  -20,  -25,  -20,  -22,  -19,  -20,
 -21,  -24,  -23,  -21,  -24,  -24,  -19,  -22,
 -19,  -21,  -18,  -19,  -23,  -26,  -21,  -17,
// queen
 -11,  -14,  -18,  -18,  -18,  -22,  -17,  -21,
 -15,  -20,  -19,  -20,  -24,  -22,  -18,  -20,
 -21,  -20,  -21,  -20,  -24,  -15,  -20,  -22,
 -21,  -24,  -28,  -22,  -23,  -24,  -24,  -11,
 -18,  -18,  -20,  -21,  -26,  -25,  -24,  -23,
 -20,  -17,  -24,  -20,  -25,  -25,  -20,  -19,
 -15,  -22,  -22,  -23,  -21,  -12,  -17,  -24,
 -15,  -22,  -19,  -21,  -18,  -27,  -17,  -25,
// king
  -6,   -8,  -10,  -12,  -13,  -11,   -7,   -7,
 -18,   -9,  -14,  -14,  -10,  -10,  -22,  -13,
 -16,  -16,  -26,  -21,  -18,  -14,  -16,  -17,
 -16,  -18,  -25,  -23,  -19,  -22,  -20,  -18,
 -18,  -21,  -12,  -21,  -25,  -22,  -15,  -16,
 -21,  -23,  -17,  -24,  -21,  -23,  -21,  -17,
 -22,  -25,  -22,  -26,  -23,  -25,  -18,  -17,
 -22,  -18,  -20,  -19,  -19,  -19,  -27,  -12,
};

constexpr unsigned kMaxSearchDepth = 64;
constexpr unsigned kMaxPlyFromRoot = kMaxSearchDepth + 32;  // Allow some extra ply since we extend depth sometimes.

struct Killers {
  Move moves[2];
  uint8_t index;
  void add(Move move) {
    moves[index] = move;
    index = (index + 1) % 2;
  }
  bool contains(Move move) const {
    return moves[0] == move || moves[1] == move;
  }
};

struct Frame {
  Killers killers;
  Move responseTo[Piece::NUM_PIECES][64];
  Move responseFrom[Piece::NUM_PIECES][64];
};

struct Thread {
  uint64_t id_;
  uint64_t multiPV_;
  unsigned depth_{1};
  std::chrono::high_resolution_clock::time_point stopTime_;
  Position position_;
  std::shared_ptr<EvaluatorInterface> evaluator_;
  std::unordered_set<Move> permittedMoves_;
  std::vector<std::pair<Move, Evaluation>> primaryVariations_;  // Contains multiPV number of best moves.
  uint64_t nodeCount_{0};
  uint64_t qNodeCount_{0};
  uint64_t nodeLimit_{(uint64_t)-1};
  Frame buffer[2];  // Empty buffer so that frames_[plyFromRoot - 2] is valid.
  Frame frames_[kMaxPlyFromRoot];

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
    nodeLimit_(other.nodeLimit_),
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

const Evaluation kQMoveOrderingPieceValue[Piece::NUM_PIECES] = {
  1000,    // NO_PIECE (means it's a check)
  100,  // PAWN
  320,  // KNIGHT
  330,  // BISHOP
  500,  // ROOK
  900,  // QUEEN
  20000 // KING
};

// The only difference between search and qsearch is that this considers NO_PIECE to have a value of zero
// since it is not a reliable indicator of a check move in normal search.
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
NegamaxResult<TURN> qsearch(Thread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, int quiescenceDepth, std::atomic<bool> *stopThinking) {
  if (IS_PRINT_QNODE) {
    std::cout << repeat("  ", plyFromRoot) << "Quiescence search called: alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " quiescenceDepth=" << quiescenceDepth << " history" << thread->position_.history_ << std::endl;
  }

  // Transposition Table probe
  TTEntry entry;
  uint64_t key = thread->position_.currentState_.hash;
  if (thread->tt_->probe(key, entry)) {
    if (entry.bound == BoundType::EXACT) {
      return NegamaxResult<TURN>(entry.bestMove, entry.value);
    } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
      return NegamaxResult<TURN>(entry.bestMove, entry.value);
    } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
      return NegamaxResult<TURN>(entry.bestMove, entry.value);
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
    return NegamaxResult<TURN>(kNullMove, kCheckmate);
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
  Threats<TURN> threats(thread->position_);
  for (ExtMove* move = moves; move < end; ++move) {
    if (move->move == entry.bestMove) {
      move->score = kMaxEval;
      continue;
    }
    assert(move->move.from < 64);
    assert(move->move.to < 64);
    Piece capturedPiece = cp2p(thread->position_.tiles_[move->move.to]);
    assert(capturedPiece < Piece::NUM_PIECES);

    move->score = kQMoveOrderingPieceValue[cp2p(move->capture)];
    move->score -= value_or_zero(
      ((threats.badForOur[move->piece] & bb(move->move.to)) > 0) && !((threats.badForOur[move->piece] & bb(move->move.from)) > 0),
      kQMoveOrderingPieceValue[move->piece]
    );
  }
  std::sort(
    moves,
    end,
    [](const ExtMove& a, const ExtMove& b) {
      return a.score > b.score;
    }
  );

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

    ColoredEvaluation<TURN> eval = -qsearch<opposite_color<TURN>()>(thread, -beta, -alpha, plyFromRoot + 1, quiescenceDepth + 1, stopThinking).evaluation;
    if (eval.value < kLongestForcedMate) {
      eval.value += 1;
    } else if (eval.value > -kLongestForcedMate) {
      eval.value -= 1;
    }
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
NegamaxResult<TURN> negamax(Thread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, std::atomic<bool> *stopThinking) {
  assert(thread->position_.turn_ == TURN);
  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Negamax called: depth=" << depth << " alpha=" << alpha.value << " beta=" << beta.value << " plyFromRoot=" << plyFromRoot << " history " << thread->position_.history_ << std::endl;
  }
  const ColoredEvaluation<TURN> originalAlpha = alpha;

  // Check for immediate cutoffs based on mate distance.
  Evaluation lowestPossibleEvaluation = kCheckmate + plyFromRoot;
  if (SEARCH_TYPE != SearchType::ROOT && lowestPossibleEvaluation >= beta.value) {
    return NegamaxResult<TURN>(kNullMove, beta.value);
  }
  Evaluation highestPossibleEvaluation = -kCheckmate - plyFromRoot;
  if (SEARCH_TYPE != SearchType::ROOT && highestPossibleEvaluation <= alpha.value) {
    return NegamaxResult<TURN>(kNullMove, alpha.value);
  }

  // Transposition Table probe
  TTEntry entry;
  const uint64_t key = thread->position_.currentState_.hash;
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
  if (thread->nodeCount_ >= thread->nodeLimit_) {
    stopThinking->store(true);
  }

  if (depth == 0) {
    if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Entering quiescence search." << std::endl;
    }
    return qsearch(thread, alpha, beta, plyFromRoot, 0, stopThinking);
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
      return NegamaxResult<TURN>(kNullMove, kCheckmate);
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

  // If we don't have a best move from the TT, we compute one with reduced depth.
  if (depth > 2 && (entry.key != key || entry.bestMove == kNullMove)) {
    NegamaxResult<TURN> result = negamax<TURN, SEARCH_TYPE>(thread, depth - 2, alpha, beta, plyFromRoot, stopThinking);
    entry.bestMove = result.bestMove;
    entry.value = result.evaluation.value;
  }

  // Add score to each move.
  const Move lastMove = SEARCH_TYPE == SearchType::ROOT ? kNullMove : thread->position_.history_.back().move;
  for (ExtMove* move = moves; move < end; ++move) {
    move->score = move->move == entry.bestMove ? 10000 : 0;
    // Prioritize captures after the TT move.
    move->score += kMoveOrderingPieceValue[cp2p(thread->position_.tiles_[move->move.to])];
    // Next prioritize the killer move(s).
    move->score += thread->frames_[plyFromRoot].killers.contains(move->move) ? 50 : 0;
    // Prioritize moves that caused a beta cutoff in a similar position, in response to a similar move.
    move->score += thread->frames_[plyFromRoot].responseTo[move->piece][lastMove.to] == move->move ? 20 : 0;
    move->score += thread->frames_[plyFromRoot].responseFrom[move->piece][lastMove.from] == move->move ? 20 : 0;

    // Prioritize moves that caused a beta cutoff on our previous move.
    move->score += thread->frames_[plyFromRoot - 2].responseTo[move->piece][lastMove.to] == move->move ? 10 : 0;
    move->score += thread->frames_[plyFromRoot - 2].responseFrom[move->piece][lastMove.from] == move->move ? 10 : 0;

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

  if (SEARCH_TYPE == SearchType::ROOT) {
    thread->primaryVariations_.clear();
  }

  NegamaxResult<TURN> bestResult(kNullMove, kMinEval);
  Move bestMoveTT = kNullMove;
  uint8_t numQuietMovesSearched = 0;
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


    ColoredEvaluation<TURN> eval(0);
    int childDepth = depth - 1;

    // Don't reduce depth for sensible captures (Elo difference: 254.7 +/- 286.2, LOS: 98.7 %)
    if (move->capture != ColoredPiece::NO_COLORED_PIECE && cp2p(move->capture) > move->piece) {
      childDepth += 1;
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
      eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, std::max(childDepth - reduction, 0), -(alpha + 1), -alpha, plyFromRoot + 1, stopThinking).evaluation;
      if (eval.value > alpha.value) {
        eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, childDepth, -beta, -alpha, plyFromRoot + 1, stopThinking).evaluation;
      }
    } else {
      // Simple, full-window, full-depth search. Used for the first move in non-root search.
      // In the root node, we only use this when multiPV==1, since we don't care about
      // the exact evaluation of moves that aren't the best move.
      eval = -negamax<opposite_color<TURN>(), SearchType::NORMAL_SEARCH>(thread, childDepth, -beta, -alpha, plyFromRoot + 1, stopThinking).evaluation;
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
        // TODO: check if this move is quiet. Probably also check if we've already added it as a killer.
        thread->frames_[plyFromRoot].killers.add(move->move);
        thread->frames_[plyFromRoot].responseTo[move->piece][lastMove.to] = move->move;
        thread->frames_[plyFromRoot].responseFrom[move->piece][lastMove.from] = move->move;
        break;
      }
    }
  }

  if (stopThinking->load()) {
    // Search was stopped externally. We cannot trust the result
    // of our for loop above, so look up the best move from the TT.
    if (thread->tt_->probe(key, entry)) {
      if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Search stopped externally. Returning TT best move." << std::endl;
      }
      return NegamaxResult<TURN>(entry.bestMove, entry.value);
    } else {
      if (IS_PRINT_NODE) {
      std::cout << repeat("  ", plyFromRoot) << "Search stopped externally. No TT entry found, returning null move." << std::endl;
      }
      return NegamaxResult<TURN>(kNullMove, 0);
    }
  }

  // Store in Transposition Table
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;

  if (IS_PRINT_NODE) {
    std::cout << repeat("  ", plyFromRoot) << "Storing in TT: depth=" << depth << " eval=" << bestResult.evaluation.value << " bound=" << bound_type_to_string(bound) << std::endl;
  }
  thread->tt_->store(
    thread->position_.currentState_.hash,
    bestMoveTT,
    depth,
    bestResult.evaluation.value,
    bound
  );

  if (IS_PRINT_NODE) {
  std::cout << repeat("  ", plyFromRoot) << "Negamax returning: bestMove=" << bestResult.bestMove.uci() << " eval=" << bestResult.evaluation.value  << " depth=" << depth << std::endl;
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
SearchResult<TURN> search(Thread* thread, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<TURN>)> onDepthCompleted, bool timeSensitive) {
  assert(thread->position_.turn_ == TURN);
  std::atomic<bool> neverStopThinking{false};
  NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(
    thread,
    1,
    /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
    /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
    /*plyFromRoot=*/0,
    &neverStopThinking  // Guarantee we always search at least depth 1 before stopping.
  );
  SearchResult<TURN> searchResult = negamax_result_to_search_result<TURN>(result, thread);
  if (onDepthCompleted != nullptr) {
    onDepthCompleted(1, searchResult);
  }
  for (int i = 2; i <= std::min(thread->depth_, kMaxSearchDepth); ++i) {
    if (stopThinking->load()) break;
    result = negamax<TURN, SearchType::ROOT>(
      thread,
      i,
      /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
      /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
      /*plyFromRoot=*/0,
      stopThinking
    );
    searchResult = negamax_result_to_search_result<TURN>(result, thread);
    if (stopThinking->load()) {
      // Primary variations may be incomplete or invalid if the search was stopped.
      // Re-run the search at depth=1 to get a valid result.
      result = negamax<TURN, SearchType::ROOT>(
        thread,
        1,
        /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
        /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
        /*plyFromRoot=*/0,
        &neverStopThinking
      );
      searchResult = negamax_result_to_search_result<TURN>(result, thread);
    }
    if (onDepthCompleted != nullptr) {
      onDepthCompleted(i, searchResult);
    }
    if (timeSensitive && (result.evaluation.value <= kLongestForcedMate || result.evaluation.value >= -kLongestForcedMate)) {
      // If we're in an actual game, stop searching deeper once we find a forced mate.
      break;
    }
  }
  return searchResult;
}

// Non-color-templated search function to be used by the UCI interface.
SearchResult<Color::WHITE> colorless_search(
  Thread* thread,
  std::atomic<bool> *stopThinking,
  std::function<void(int, SearchResult<Color::WHITE>)> onDepthCompleted,
  bool timeSensitive
);

// Convenience function to search programmatically without needing to specify color or create a thread.
SearchResult<Color::WHITE> search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable* tt);

}  // namespace ChessEngine

#endif  // SEARCH_H
