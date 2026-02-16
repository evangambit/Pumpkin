#ifndef SEARCH_H
#define SEARCH_H

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
#include "negamax.h"

namespace ChessEngine {

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
  thread->tt_->new_search();
  assert(thread->position_.turn_ == TURN);
  std::atomic<bool> neverStopThinking{false};
  NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(
    thread,
    1,
    /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
    /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
    /*plyFromRoot=*/0,
    &thread->frames_[0],
    &neverStopThinking  // Guarantee we always search at least depth 1 before stopping.
  );
  if (result.bestMove == kNullMove) {
    std::cout << "Error: Search did not find a move." << std::endl;
    exit(1);
  }
  SearchResult<TURN> searchResult = negamax_result_to_search_result<TURN>(result, thread);
  SearchResult<TURN> lastResult = searchResult;
  if (onDepthCompleted != nullptr) {
    onDepthCompleted(1, searchResult);
  }
  for (int i = 2; i <= std::min(thread->depth_, kMaxSearchDepth); ++i) {
    if (stopThinking->load()) {
      break;
    }
    result = negamax<TURN, SearchType::ROOT>(
      thread,
      i,
      /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
      /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
      /*plyFromRoot=*/0,
      &thread->frames_[0],
      stopThinking
    );
    if (result.bestMove == kNullMove) {
      std::cout << "Error: Search did not find a move." << result << std::endl;
      exit(1);
    }
    searchResult = negamax_result_to_search_result<TURN>(result, thread);
    if (stopThinking->load()) {
      // Primary variations may be incomplete or invalid if the search was stopped.
      // Fallback to the last completed search result, which is guaranteed to be valid.
      searchResult = lastResult;
    }
    lastResult = searchResult;
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
