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
struct MoveCache {
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

enum NodeType {
  PV_NODE,
  CUT_NODE,
  ALL_NODE
};

/** Ply-specific information. */
struct Frame {
  MoveCache killers;
  Move responseTo[Piece::NUM_PIECES][64];
  Move responseFrom[Piece::NUM_PIECES][64];
  uint64_t hash;
  bool inCheck;
  Evaluation staticEval;
  Move excludedMove = kNullMove;
  NodeType nodeType;
};

struct HistoryEntry {
  int32_t value{0};

  int32_t score() const {
    return value / 64;
  }

  void update(int depth) {
    const int32_t bonus = std::min(depth * depth, 400);
    value += bonus - value * std::abs(bonus) / 16384;
  }
};


struct GoCommand {
  GoCommand()
  : depthLimit(kMaxSearchDepth), nodeLimit(-1), timeLimitMs(-1),
  wtimeMs(0), btimeMs(0), wIncrementMs(0), bIncrementMs(0), movesUntilTimeControl(-1), makeBestMove(false), isPondering(false) {}

  Position pos;

  size_t depthLimit;
  uint64_t nodeLimit;
  uint64_t timeLimitMs;
  std::unordered_set<Move> moves;

  uint64_t wtimeMs;
  uint64_t btimeMs;
  uint64_t wIncrementMs;
  uint64_t bIncrementMs;
  uint64_t movesUntilTimeControl;

  // If true, the best (found) move is made after the command finishes.
  bool makeBestMove;
  bool isPondering;
};


static constexpr unsigned kNumSearchManagerCounters = 32768;
static constexpr unsigned kNumSearchManagerLocks = 256;
struct SearchManager {
  uint8_t counters[kNumSearchManagerCounters];
  SpinLock locks[kNumSearchManagerLocks];
  SearchManager() {
    std::fill_n(counters, kNumSearchManagerCounters, 0);
  }
  bool should_start_searching(uint64_t hash) {
    size_t idx = hash % kNumSearchManagerCounters;
    SpinLock& lock = locks[hash % kNumSearchManagerLocks];
    lock.lock();
    bool r = counters[idx] == 0;
    if (r) {
      counters[idx] += 1;
    }
    lock.unlock();
    return r;
  }
  void finished_searching(uint64_t hash) {
    size_t idx = hash % kNumSearchManagerCounters;
    SpinLock& lock = locks[hash % kNumSearchManagerLocks];
    lock.lock();
    counters[idx] -= 1;
    lock.unlock();
  }
};

struct SearchHyperParams {
  FixedPoint<int32_t, 8> lmr_pv_a = 0.4;
  FixedPoint<int32_t, 8> lmr_pv_b = 0.3;
  FixedPoint<int32_t, 8> lmr_null_a = 0.9;

  // +0.02 vs -0.02 : +3201-3301=4990  -0.004±0.003  p=0.163  (5746/10000 total)
  //  +0.2 vs -0.2  :  +727 -705=1010   0.005±0.007  p=0.530  (1221/10000 total)
  //  +1.9 vs -1.9  :  +1088-976=1430  0.016±0.006
  //  +0.0 vs +0.8  :  +1088-976=1430  0.016±0.006
  //  +0.0 vs +0.4  :  +0.0016 +/- 0.0026
  FixedPoint<int32_t, 8> lmr_null_b = 0.4;

  // Time Control: 5+0.1s
  // 20 vs 35: +723-729=1036  -0.001±0.006  p=0.828
  // 35 vs 50: +583-574= 843   0.002±0.006  p=0.713
  // 35 vs 65: + 94- 72=  98   0.042±0.016  p=0.010
  int singular_margin = 50;

  // 15 vs 25: +5900-6060=8040  -0.004±0.003  p=0.112  (10000/10000 total)
  // 25 vs 35: +1977-1978=2849  -0.000±0.004  p=0.986  (3402/10000 total)
  // 25 vs 55:  +656- 572= 972   0.019±0.007  p=0.009  (1100/10000 total)
  int razoring_margin = 20;

  // 15 vs 25:  +779- 881=1244  -0.018±0.007  p=0.009  (1452/10000 total)
  // 25 vs 35: +5317-5207=7144   0.003±0.003  p=0.265  (8834/10000 total)
  // 35 vs 45: +1533-1477=2138   0.005±0.005  p=0.293  (2574/10000 total)
  // 45 vs 55:    +255-202=347   0.033±0.013  p=0.009  (402/10000 total)
  int futility_margin = 30;

  // Experiment results:
  //  1 Win50     :     7.6    3.7  1233.5    2400    51
  //  2 Win25     :    -0.5    3.8  1198.0    2400    50
  //  3 Win100    :    -1.1    3.7  1195.0    2400    50
  //  4 Old       :    -6.0    3.7  1173.5    2400    49
  int aspiration_window = 50;

  int max_eval_for_null_window_search = 1000;

  int null_move_pruning_depth_reduction = 5;
};

/**
 * State for a single search which is shared across the 1+ threads performing that search.
 */
struct SharedSearchThreadState {
  SharedSearchThreadState(const GoCommand& command, unsigned multiPV, unsigned numThreads, bool isTimeSensitive, std::chrono::high_resolution_clock::time_point stopTime, TranspositionTable* tt)
  : tt(tt), searchManager(std::make_unique<SearchManager>()), stopTime(stopTime), permittedMoves(command.moves), multiPV(multiPV), numThreads(numThreads), isTimeSensitive(isTimeSensitive), nodeLimit(command.nodeLimit), depthLimit(command.depthLimit), timeLimitMs(command.timeLimitMs), isPondering(command.isPondering) {

  }

  SearchHyperParams search_hyper_params;

  // This pointer should be considered non-owning. The TranspositionTable should created and
  // managed elsewhere since it should be shared across threads and searches.
  TranspositionTable* tt;
  std::unique_ptr<SearchManager> searchManager;

  const unsigned multiPV;
  const unsigned numThreads;

  // Set to true if we're playing with time controls so that (e.g.) stopping early may be advantageous.
  // Is *not* set to true for "go movetime 1000", since there is no advantage to returning early.
  const bool isTimeSensitive;

  std::chrono::high_resolution_clock::time_point stopTime;
  const std::unordered_set<Move> permittedMoves;
  const unsigned depthLimit{1};
  const uint64_t nodeLimit;
  const uint64_t timeLimitMs;
  std::atomic<bool> isPondering;

  // TODO: fix race condition to stopTime (UCI thread tries to write to it too).
  void ponderHit() {
    isPondering.store(false);
    if (isTimeSensitive) {
      stopTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(timeLimitMs);
    } else if (timeLimitMs != (uint64_t)-1) {
      stopTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(timeLimitMs);
    }
  }
};



/**
  * Thread-specific information. e.g. every thread has its own nodeCount_, position, etc.
  */
struct SearchThread {
  const uint64_t id_;
  Position position_;  // Note: position contains a pointer to the evaluator.
  std::vector<std::pair<Move, Evaluation>> primaryVariations_;  // Contains multiPV number of best moves.
  uint64_t nodeCount_{0};
  uint64_t qNodeCount_{0};
  Frame frames_[kMaxPlyFromRoot + 4];  // Root search starts at frames_[4] so frame[-4] lookbacks stay in-bounds.
  HistoryEntry quietHistory_[Piece::NUM_PIECES][64];
  HistoryEntry captureHistory_[Piece::NUM_PIECES][Piece::NUM_PIECES][64];

  std::shared_ptr<SharedSearchThreadState> shared_;

  SearchThread(
    uint64_t id,
    const Position& pos,
    std::shared_ptr<SharedSearchThreadState> shared
  ) : id_(id), position_(pos), shared_(shared) {
    std::memset(frames_, 0, sizeof(frames_));
    std::memset(quietHistory_, 0, sizeof(quietHistory_));
    std::memset(captureHistory_, 0, sizeof(captureHistory_));
  }
  
  SearchThread(const SearchThread& other)
  : id_(other.id_),
    position_(other.position_),
    primaryVariations_(other.primaryVariations_),
    nodeCount_(other.nodeCount_),
    qNodeCount_(other.qNodeCount_),
    shared_(other.shared_) {
      std::memcpy(frames_, other.frames_, sizeof(frames_));
      std::memcpy(quietHistory_, other.quietHistory_, sizeof(quietHistory_));
      std::memcpy(captureHistory_, other.captureHistory_, sizeof(captureHistory_));
    }

  Frame* root_frame() {
    return &frames_[4];
  }

  const Frame* root_frame() const {
    return &frames_[4];
  }

  ptrdiff_t ply_from_root(const Frame* frame) const {
    return frame - root_frame();
  }

  bool should_stop() const {
    if (shared_->isPondering.load()) {
      return nodeCount_ >= shared_->nodeLimit;
    }
    return std::chrono::high_resolution_clock::now() >= shared_->stopTime || nodeCount_ >= shared_->nodeLimit;
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

inline int qsearch_tt_depth(int quiescenceDepth) {
  return -quiescenceDepth;
}

template<Color TURN>
NegamaxResult<TURN> qsearch(SearchThread* thread, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, int quiescenceDepth, Frame *frame, std::atomic<bool> *stopThinking) {
  frame->hash = thread->position_.currentState_.hash;

  // This can happen when we've already found a checkmate in a previous sibling/ancestor node.
  if (alpha >= beta) {
    return NegamaxResult<TURN>(kNullMove, beta);
  }

  // Prevent overflowing the frame array (and the call stack) - return static eval if we've gone too deep
  if (quiescenceDepth >= kMaxQuiescenceDepth || thread->ply_from_root(frame) >= kMaxPlyFromRoot - 1) {
    Threats threats;
    create_threats(thread->position_.pieceBitboards_, thread->position_.colorBitboards_, &threats);
    return NegamaxResult<TURN>(kNullMove, evaluate<TURN>(thread->position_.evaluator_, thread->position_, threats, plyFromRoot, alpha, beta).clamp_(alpha, beta));
  }

  static constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();

  // Check if draw by repetition. is_3fold_repetition short-circuits when the
  // last move was a capture or pawn move, so this is near-free in the common
  // qsearch case (captures). It still correctly detects repetitions through
  // check sequences.
  const bool isThreefoldDraw = thread->position_.is_3fold_repetition(plyFromRoot);
  // Check if draw by fifty-move rule. We have to exclude positions where the king is in check
  // since it may be checkmate.
  const bool isFiftyMoveDraw = thread->position_.is_fifty_move_rule();
  if (isThreefoldDraw || (isFiftyMoveDraw && !frame->inCheck)) {
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(alpha, beta));
  }

  // Transposition Table probe
  TTEntry entry{0, kNullMove, 0, 0, BoundType::EXACT, 0};
  uint64_t key = thread->position_.currentState_.hash;
  const int ttDepth = qsearch_tt_depth(quiescenceDepth);
  if (thread->shared_->tt->probe(key, entry)) {
    if (entry.depth >= ttDepth) {
      if (entry.bound == BoundType::EXACT) {
        return NegamaxResult<TURN>(entry.bestMove, ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta));
      } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
        return NegamaxResult<TURN>(entry.bestMove, beta);
      } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
        return NegamaxResult<TURN>(entry.bestMove, alpha);
      }
    }
  } else {
    entry.bestMove = kNullMove;
  }

  const ColoredEvaluation<TURN> originalAlpha = alpha;

  thread->nodeCount_++;
  thread->qNodeCount_++;

  if ((thread->nodeCount_ & 1023) == 0) {
    if (thread->should_stop()) {
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

  if (moves == end && frame->inCheck) {
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate).clamp_(originalAlpha, beta));
  }
  // Now that we know it's not checkmate, we can re-check for fifty-move draw.
  if (isFiftyMoveDraw) {
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(originalAlpha, beta));
  }

  Threats threats;
  create_threats(thread->position_.pieceBitboards_, thread->position_.colorBitboards_, &threats);
  frame->staticEval = evaluate<TURN>(thread->position_.evaluator_, thread->position_, threats, plyFromRoot, alpha, beta).value;
  NegamaxResult<TURN> bestResult(kNullMove, frame->inCheck ? alpha : ColoredEvaluation<TURN>(frame->staticEval).clamp_(alpha, beta));
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
  const Move lastMove = thread->position_.history_.size() > 0 ? thread->position_.history_.back().move : kNullMove;
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
  }
  std::sort(
    moves,
    end,
    [](const ExtMove& a, const ExtMove& b) {
      return a.score > b.score;
    }
  );

  for (ExtMove* move = moves; move < end; ++move) {
    if (move->score <= 0 && !frame->inCheck) {
      // Don't consider moves that lose material according to move ordering heuristic.
      continue;
    }
    make_move<TURN>(&thread->position_, move->move);

    // Move generation can sometimes generate illegal en passant moves.
    static constexpr ColoredPiece enemyKing = coloredPiece<opposite_color<TURN>(), Piece::KING>();
    const bool moveGivesCheck = can_enemy_attack<opposite_color<TURN>()>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[enemyKing])
    );
    (frame + 1)->inCheck = moveGivesCheck;

    static constexpr ColoredPiece moverKing = coloredPiece<TURN, Piece::KING>();
    if (can_enemy_attack<TURN>(
      thread->position_,
      lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
    )) {
      // Need this check because of en passant captures into check.
      // e.g. b5c6 in position 8/1k6/6R1/KPpr4/8/8/8/8 w - c6 0 62
      undo<TURN>(&thread->position_);
      continue;
    }

    ColoredEvaluation<TURN> eval = to_parent_eval(qsearch<opposite_color<TURN>()>(thread, to_child_eval(beta), to_child_eval(alpha), plyFromRoot + 1, quiescenceDepth + 1, frame + 1, stopThinking).evaluation);
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

  if (stopThinking->load()) {
    return bestResult;
  }

  // Store in Transposition Table
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;

  thread->shared_->tt->store(
    thread->position_.currentState_.hash,
    bestResult.bestMove,
    ttDepth,
    bestResult.evaluation.value,
    bound
  );

  return bestResult;
}

template<Color TURN>
inline bool sanity_check(const Position& pos, Move move) {
  if (TURN == Color::WHITE) {
    return pos.tiles_[move.from] >= ColoredPiece::WHITE_PAWN && pos.tiles_[move.from] <= ColoredPiece::WHITE_KING && is_move_feasible(cp2p(pos.tiles_[move.from]), move);
  } else {
    return pos.tiles_[move.from] >= ColoredPiece::BLACK_PAWN && pos.tiles_[move.from] <= ColoredPiece::BLACK_KING && is_move_feasible(cp2p(pos.tiles_[move.from]), move);
  }
}

/**
 * Note: if you set stopThinking to true, there is no guarantee that this will return a sensible/valid result.
 * In practice, you will likely want to re-search with depth=1 and stopThinking=false to get a valid move.
 */
template<Color TURN, SearchType SEARCH_TYPE, bool IS_MULTITHREADED>
NegamaxResult<TURN> negamax(SearchThread* thread, int depth, ColoredEvaluation<TURN> alpha, ColoredEvaluation<TURN> beta, int plyFromRoot, Frame *frame, std::atomic<bool> *stopThinking) {
  assert(thread->position_.turn_ == TURN);
  const uint64_t key = thread->position_.currentState_.hash;
  frame->hash = key;
  const ColoredEvaluation<TURN> originalAlpha = alpha;

  // Because of how we handle checkmate values, this condition is basically
  // how we avoid looking for mate-in-5 if we already found mate-in-4.
  if (alpha >= beta) {
    return NegamaxResult<TURN>(kNullMove, beta);
  }

  if (thread->ply_from_root(frame) >= kMaxPlyFromRoot - 1) {
    const auto r = qsearch<TURN>(thread, alpha, beta, plyFromRoot, kMaxQuiescenceDepth, frame, stopThinking);
    return r;
  }

  if (depth == 0) {
    assert(SEARCH_TYPE != SearchType::ROOT);
    return qsearch(thread, alpha, beta, plyFromRoot, 0, frame, stopThinking);
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
    frame->nodeType = NodeType::PV_NODE;
  }

  if (SEARCH_TYPE != SearchType::ROOT && stopThinking->load()) {
    return NegamaxResult<TURN>(kNullMove, originalAlpha);
  }

  // Check if draw by repetition or insufficient material.
  const bool isThreefoldDraw = thread->position_.is_3fold_repetition(plyFromRoot);
  const bool isFiftyMoveDraw = thread->position_.is_fifty_move_rule();
  bool isDraw = isThreefoldDraw;
  isDraw |= isFiftyMoveDraw && !frame->inCheck;
  isDraw |= thread->position_.is_material_draw();
  if (isDraw) {
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(originalAlpha, beta));
  }

  // Transposition Table probe
  TTEntry entry{0, kNullMove, 0, 0, BoundType::EXACT, 0};
  if (frame->excludedMove == kNullMove && thread->shared_->tt->probe(key, entry) && sanity_check<TURN>(thread->position_, entry.bestMove)) {
    if (entry.depth >= depth) {
      // Only use TT cutoffs in non-PV nodes (NULL_WINDOW_SEARCH).
      // In PV nodes (NORMAL_SEARCH), we only use the TT for move ordering
      // to avoid graph history interaction (GHI) bugs where path-dependent
      // evaluations (e.g. from repetition draws) are incorrectly reused.
      if (SEARCH_TYPE == SearchType::NULL_WINDOW_SEARCH) {
        if (entry.bound == BoundType::EXACT) {
          return NegamaxResult<TURN>(entry.bestMove, ColoredEvaluation<TURN>(entry.value).clamp_(alpha, beta));
        } else if (entry.bound == BoundType::LOWER && entry.value >= beta.value) {
          return NegamaxResult<TURN>(entry.bestMove, beta);
        } else if (entry.bound == BoundType::UPPER && entry.value <= alpha.value) {
          return NegamaxResult<TURN>(entry.bestMove, alpha);
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
    if (thread->should_stop()) {
      stopThinking->store(true);
    }
  } else if (thread->nodeCount_ >= thread->shared_->nodeLimit) {
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
    if (!thread->shared_->permittedMoves.empty()) {
      ExtMove* writePtr = moves;
      for (ExtMove* move = moves; move < end; ++move) {
        if (thread->shared_->permittedMoves.count(move->move) > 0) {
          *writePtr++ = *move;
        }
      }
      end = writePtr;
    }
  }

  if (moves == end) {
    if (frame->inCheck) {
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate).clamp_(originalAlpha, beta));
    } else {
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(Evaluation(kDraw)).clamp_(originalAlpha, beta));
    }
  }

  // Now that we know it's not checkmate, we can re-check for fifty-move draw.
  if (isFiftyMoveDraw) {
    return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(originalAlpha, beta));
  }

  if (depth > 4 && entry.bestMove == kNullMove) {
    if (SEARCH_TYPE != SearchType::NULL_WINDOW_SEARCH) {
      // Internal Iterative Deepening.
      // If we don't have a best move from the TT, we compute one with reduced depth.
      NegamaxResult<TURN> result = negamax<TURN, SEARCH_TYPE, IS_MULTITHREADED>(thread, depth - 4, alpha, beta, plyFromRoot, frame, stopThinking);
      entry.bestMove = result.bestMove;
      entry.value = result.evaluation.value;
    } else {
      // However, for null window searches we simply reduce the depth by 2 to avoid the
      // significant overhead of a shallow search.
      depth -= 2;
    }
  }

  // Add score to each move.
  Threats threats;
  create_threats(thread->position_.pieceBitboards_, thread->position_.colorBitboards_, &threats);
  // We otherwise never call evaluate in interior nodes, but it behooves us to keep the accumulator
  // up-to-date so our children/grandchildren can benefit from it.
  frame->staticEval = evaluate<TURN>(thread->position_.evaluator_, thread->position_, threats, plyFromRoot, alpha, beta).value;

  const auto& searchHyperParams = thread->shared_->search_hyper_params;

  // Razoring.
  #if EVAL_AGNOSTIC == 0
  if (SEARCH_TYPE != SearchType::ROOT && depth <= 2 && frame->staticEval < alpha.value - searchHyperParams.razoring_margin * depth * depth) {
    const auto r = qsearch<TURN>(thread, alpha, beta, plyFromRoot, 0, frame, stopThinking);
    // We should check "r.evaluation <= alpha" here, but omitting the check
    // seems to perform better in practice... somehow. Changing the above
    // search to a null-window search also makes us perform worse, which is
    // also quite counterintuitive -- surely a null-window search is pure savings
    // on a move that we're trying to prove bad... right?
    return r;
  }

  // 60/40: +164-212=356  -0.033±0.012  p=0.009  (366/10000 total)
  // 40/60: +226-235=405  -0.005±0.011  p=0.646  (433/10000 total)
  // 40/40: +218-208=344  0.006±0.012  p=0.603  (385/10000 total)
  // 20/20: +200-241=393  -0.025±0.012  p=0.038  (417/10000 total)

  // Reverse futility pruning (+29.6 ± 2.7)
  if (SEARCH_TYPE == SearchType::NULL_WINDOW_SEARCH && depth == 1 && frame->staticEval > beta.value + searchHyperParams.futility_margin && !frame->inCheck && std::abs(beta.value) < searchHyperParams.max_eval_for_null_window_search) {
    const auto r = NegamaxResult<TURN>(kNullMove, beta);
    return r;
  }

  // Null move pruning.
  // This is roughly equivalent to having twice as much time.
  //  # PLAYER       :  RATING  ERROR  POINTS  PLAYED   (%)
  //  1 nmp-slow     :    64.1    5.5  2963.5    4800    62
  //  2 nmp-fast     :     2.3    5.4  2420.5    4800    50
  //  3 main-slow    :    -4.1    5.5  2364.0    4800    49
  //  4 main-fast    :   -62.3    5.4  1852.0    4800    39
  const int myPieceCount = std::popcount(thread->position_.colorBitboards_[TURN] & ~thread->position_.pieceBitboards_[coloredPiece<TURN, Piece::PAWN>()]);
  if (SEARCH_TYPE == SearchType::NULL_WINDOW_SEARCH && !frame->inCheck && myPieceCount > 1 && depth > 0) {
    const int reducedDepth = std::max(0, depth - searchHyperParams.null_move_pruning_depth_reduction);
    make_nullmove<TURN>(&thread->position_);
    (frame + 1)->inCheck = false;
    (frame + 1)->nodeType = NodeType::ALL_NODE;
    ColoredEvaluation<TURN> r = to_parent_eval(negamax<opposite_color<TURN>(), SearchType::NULL_WINDOW_SEARCH, IS_MULTITHREADED>(
      thread, reducedDepth, to_child_eval(beta), to_child_eval(beta - 1), plyFromRoot + 1, frame + 1, stopThinking
    ).evaluation);
    undo_nullmove<TURN>(&thread->position_);
    if (r >= beta) {
      return NegamaxResult<TURN>(kNullMove, beta);
    }
  }
  #endif  // EVAL_AGNOSTIC

  const Move lastMove = (SEARCH_TYPE == SearchType::ROOT || thread->position_.history_.empty())
    ? kNullMove
    : thread->position_.history_.back().move;
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

  const Bitboard theirTargets = TURN == Color::WHITE ? threats.blackTargets : threats.whiteTargets;
  const Bitboard theirPawns = thread->position_.pieceBitboards_[coloredPiece<opposite_color<TURN>(), Piece::PAWN>()];
  const Bitboard theirPieces = TURN == Color::WHITE ? thread->position_.colorBitboards_[Color::BLACK] : thread->position_.colorBitboards_[Color::WHITE];
  const Bitboard theirUndefendedPieces = (theirPieces & ~theirPawns) & ~theirTargets;

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
      move->score += thread->quietHistory_[move->piece][move->move.to].score();
    } else {
      move->score += thread->captureHistory_[move->piece][cp2p(thread->position_.tiles_[move->move.to])][move->move.to].score();
    }

    // Penalize non-capture moves that move to a defended square.
    move->score -= value_or_zero(
      ((threats.badForOur<TURN>(move->piece) & bb(move->move.to)) > 0)
      &&
      move->capture == ColoredPiece::NO_COLORED_PIECE
    , kMoveOrderingPieceValue[move->piece]);
    // Bonus for moving a piece that is under attack.
    move->score += value_or_zero(
      ((threats.badForOur<TURN>(move->piece) & bb(move->move.from)) > 0)
    , kMoveOrderingPieceValue[move->piece] / 2);

    // Prioritize moves that caused a beta cutoff in a similar position, in response to a similar move.
    // NOTE: this is different than conventional response-move heuristic since we use move->piece, not lastMove's piece.
    // Curiously, switching to the standard response-move heuristic doesn't seem to help at all (-0.004±0.006).
    move->score += frame->responseTo[move->piece][lastMove.to] == move->move ? 25 : 0;
    move->score += frame->responseFrom[move->piece][lastMove.from] == move->move ? 20 : 0;
    move->score += (frame - 2)->responseTo[move->piece][lastMove.to] == move->move ? 15 : 0;
    move->score += (frame - 2)->responseFrom[move->piece][lastMove.from] == move->move ? 10 : 0;

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

  const Bitboard aheadOfTheirPawns = ((TURN == Color::WHITE) ? 
    southFill(theirPawns) : northFill(theirPawns)) & ~theirPawns;
  const Bitboard ourPassedPawnMask = ~(fatten(aheadOfTheirPawns));

  // We use kMinEval instead of alpha so that we still get a best move, even if all moves fail low.
  // This is helpful for probing the TT to try and understand why we got a cutoff. This gives a bonus
  // of +0.087±0.032 over using alpha.
   NegamaxResult<TURN> bestResult(kNullMove, ColoredEvaluation<TURN>(kMinEval));
  int numLegalMoves = 0;
  ExtMove deferredMoves[kMaxNumMoves];
  ExtMove *deferredMovesEnd = deferredMoves;

  ExtMove *activeMoves = moves;
  ExtMove *activeEnd = end;
  int moveIndex = -1;
  for (int isDeferred = 0; isDeferred < (IS_MULTITHREADED ? 2 : 1); ++isDeferred) {
    for (ExtMove* move = activeMoves; move != activeEnd; ++move) {
      static constexpr ColoredPiece enemyKing = coloredPiece<opposite_color<TURN>(), Piece::KING>();
      assert((thread->position_.pieceBitboards_[enemyKing] & bb(move->move.to)) == 0);

      if (move->move == frame->excludedMove) {
        continue;
      }

      // Singular extension: if the TT's move is significantly better than alternatives.
      // We use "entry.bound != BoundType::UPPER" since an upperbound means we never actually
      // did a thorough examination of all moves, so it is likely that the TT's "best move" isn't
      // actually the best.
      bool isSingular = false;
      #if EVAL_AGNOSTIC == 0
      if (depth > 4 && move->move == entry.bestMove && entry.depth >= depth - 3 && entry.bound != BoundType::UPPER && frame->excludedMove == kNullMove && !alpha.is_mating()) {
        frame->excludedMove = move->move;
        auto r = negamax<TURN, SearchType::NULL_WINDOW_SEARCH, IS_MULTITHREADED>(
          thread,
          (depth - 1) / 2,  // Stolen from Stockfish.
          ColoredEvaluation<TURN>(entry.value - searchHyperParams.singular_margin - 1),
          ColoredEvaluation<TURN>(entry.value - searchHyperParams.singular_margin),
          plyFromRoot,
          frame,
          stopThinking
        );
        frame->excludedMove = kNullMove;
        isSingular = r.evaluation.value < entry.value - searchHyperParams.singular_margin;
      }
      #endif  // EVAL_AGNOSTIC

      make_move<TURN>(&thread->position_, move->move);
      ++moveIndex;
      
      const bool areWeInCheck = can_enemy_attack<TURN>(
        thread->position_,
        lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[moverKing])
      );
      if (areWeInCheck) {
        // Need this check because of en passant captures into check.
        // e.g. b5c6 in position 8/1k6/6R1/KPpr4/8/8/8/8 w - c6 0 62
        undo<TURN>(&thread->position_);
        continue;
      }

      // All threads proceed to the first child, but defer children that are already being searched
      // by another thread.
      bool claimedPosition = false;
      const uint64_t childHash = thread->position_.currentState_.hash;
      if (IS_MULTITHREADED) {
        if (depth > 1 && !isDeferred && move != moves) {
          if (!thread->shared_->searchManager->should_start_searching(childHash)) {
            *deferredMovesEnd++ = *move;
            undo<TURN>(&thread->position_);
            continue;
          }
          claimedPosition = true;
        }
      }

      ++numLegalMoves;
      const bool moveGivesCheck = can_enemy_attack<opposite_color<TURN>()>(
        thread->position_,
        lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[enemyKing])
      );
      (frame + 1)->inCheck = moveGivesCheck;

      ColoredEvaluation<TURN> eval;

      // Don't reduce depth for sensible captures (Elo difference: 254.7 +/- 286.2, LOS: 98.7 %)
      const bool isGoodCapture = move->capture != ColoredPiece::NO_COLORED_PIECE && cp2p(move->capture) > move->piece;
      // Also don't reduce depth for safe passed pawn pushes.
      const bool isSafePassedPawnPush = move->piece == Piece::PAWN && (ourPassedPawnMask & ~theirTargets & bb(move->move.to)) > 0;
      const bool isSack = !!(threats.badForOur<TURN>(move->piece) & bb(move->move.to));
      const bool isQuiet = (
        (move->capture == ColoredPiece::NO_COLORED_PIECE) && (move->move.moveType != MoveType::PROMOTION)
      );

      // TODO: we can probably remove the "not checkmating" check here, but we need to be careful since null window bounds,
      // as they are currently written, can be equal! If you want to remove the "not checkmating" condition, you should test
      // with
      // $ ./uci "position fen r5k1/3Q1p2/2p3pp/4b3/p7/P1P1q3/1rBR2bP/1K1R4 w - - 0 26 moves b1a1 e3c3" "go depth 4" "lazyquit"
      const int childDepth = depth - 1 + (isSingular ? 1 : 0);
      if (move->move != moves[0].move && (SEARCH_TYPE != SearchType::ROOT || thread->shared_->multiPV == 1) && alpha.value > kLongestForcedMate && alpha.value < -kLongestForcedMate) {
        #ifndef NO_LMR
          const auto a = SEARCH_TYPE == NULL_WINDOW_SEARCH ? searchHyperParams.lmr_null_a : searchHyperParams.lmr_pv_a;
          const auto b = SEARCH_TYPE == NULL_WINDOW_SEARCH ? searchHyperParams.lmr_null_b : searchHyperParams.lmr_pv_b;
          int lateMoveReduction = (a * kLnLookup[childDepth] * kLnLookup[moveIndex] + b).floorToInt();
          lateMoveReduction -= isGoodCapture ? 1 : 0;
          lateMoveReduction -= isSafePassedPawnPush ? 1 : 0;
          lateMoveReduction += isSack ? 1 : 0;
          lateMoveReduction -= frame->killers.contains(move->move) ? 1 : 0;
          // TODO: extend attacking trapped pieces?
          const int reducedChildDepth = std::max(childDepth - std::max(0, lateMoveReduction), 0);
        #else
          const int reducedChildDepth = childDepth;
        #endif

        if (SEARCH_TYPE != SearchType::NULL_WINDOW_SEARCH) {
          // We are entering a null window search from a PV node, so the child node is a cut node.
          (frame + 1)->nodeType = NodeType::CUT_NODE;
        } else {
          // TODO: The first k children of a CUT_NODE are ALL_NODEs, but the rest should probably be CUT_NODEs.
          (frame + 1)->nodeType = (frame->nodeType == NodeType::CUT_NODE ? NodeType::ALL_NODE : NodeType::CUT_NODE);
        }
        eval = to_parent_eval(negamax<opposite_color<TURN>(), SearchType::NULL_WINDOW_SEARCH, IS_MULTITHREADED>(thread, reducedChildDepth, to_child_eval(alpha + 1), to_child_eval(alpha), plyFromRoot + 1, frame + 1, stopThinking).evaluation);
        if (eval.value > alpha.value) {
          constexpr SearchType searchType = SEARCH_TYPE == SearchType::ROOT ? SearchType::NORMAL_SEARCH : SEARCH_TYPE;
          eval = to_parent_eval(negamax<opposite_color<TURN>(), searchType, IS_MULTITHREADED>(thread, childDepth, to_child_eval(beta), to_child_eval(alpha), plyFromRoot + 1, frame + 1, stopThinking).evaluation);
        }
      } else {
        // Simple, full-window, full-depth search. Used for the first move in non-root search.
        // In the root node, we use this when multiPV==1, since we don't care about the exact
        // evaluation of moves that aren't the best move.
        static constexpr SearchType firstMoveSearchType = SEARCH_TYPE == SearchType::ROOT ? SearchType::NORMAL_SEARCH : SEARCH_TYPE;
        if (SEARCH_TYPE != SearchType::NULL_WINDOW_SEARCH) {
          (frame + 1)->nodeType = NodeType::PV_NODE;
        } else {
          (frame + 1)->nodeType = (frame->nodeType == NodeType::CUT_NODE ? NodeType::ALL_NODE : NodeType::CUT_NODE);
        }
        eval = to_parent_eval(negamax<opposite_color<TURN>(), firstMoveSearchType, IS_MULTITHREADED>(thread, childDepth, to_child_eval(beta), to_child_eval(alpha), plyFromRoot + 1, frame + 1, stopThinking).evaluation);
      }

      if (claimedPosition) {
        thread->shared_->searchManager->finished_searching(childHash);
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
          if (thread->primaryVariations_.size() >= thread->shared_->multiPV) {
            alpha = ColoredEvaluation<TURN>(thread->primaryVariations_[thread->shared_->multiPV - 1].second);
            if (thread->primaryVariations_.size() > thread->shared_->multiPV) {
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
            thread->quietHistory_[move->piece][move->move.to].update(depth);
          } else {
            thread->captureHistory_[move->piece][cp2p(move->capture)][move->move.to].update(depth);
          }
          frame->killers.add(move->move);
          frame->responseTo[move->piece][lastMove.to] = move->move;
          frame->responseFrom[move->piece][lastMove.from] = move->move;
          deferredMovesEnd = deferredMoves;  // Skip the deferred iteration by setting the number of deferred moves to 0.
          break;
        }
      }
    }
    activeMoves = deferredMoves;
    activeEnd = deferredMovesEnd;
  }

  if (numLegalMoves == 0) {
    // It is possible to have no legal moves if the only pseudo-legal moves is
    // an illegal enpassant. Example:
    // 5r2/3p4/2rp1p2/3K1Ppr/3p3k/2bb4/8/8 w - - 0 2
    // This can result in stalemate (above example) or checkmate (though it would
    // have to be a discovered check -- due to geometric constraints and the way
    // we successfully prune out most illegal moves, the pawn cannot be giving a
    // check itself, while also having enpassant be illegal).
    if (frame->inCheck) {
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kCheckmate).clamp_(originalAlpha, beta));
    } else {
      return NegamaxResult<TURN>(kNullMove, ColoredEvaluation<TURN>(kDraw).clamp_(originalAlpha, beta));
    }
    // TODO (low priority): when en passant is illegal, we still consider is "legal" for the purpose of 3-fold repetition detection,
    // meaning we may believe a position is not drawable when it actually is.
  }

  if (stopThinking->load()) {
    // Search was stopped externally. We cannot trust the result
    // of our for loop above, so we return early to avoid writing
    // an inaccurate result to the transposition table.
    return bestResult;
  }

  // Store in Transposition Table
  BoundType bound = BoundType::EXACT;
  if (bestResult.evaluation <= originalAlpha) bound = BoundType::UPPER;
  else if (bestResult.evaluation >= beta) bound = BoundType::LOWER;

  bestResult.evaluation.clamp_(originalAlpha, beta);

  if (frame->excludedMove == kNullMove) {
    thread->shared_->tt->store(
      thread->position_.currentState_.hash,
      bestResult.bestMove,
      depth,
      bestResult.evaluation.value,
      bound
    );
  }

  return bestResult;
}

}  // namespace ChessEngine

#endif  // NEGAMAX_H
