#ifndef SEARCH_H
#define SEARCH_H

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_set>

#include "../game/Position.h"
#include "../game/Utils.h"
#include "../game/movegen/movegen.h"
#include "../game/Threats.h"
#include "../eval/Evaluator.h"
#include "../eval/ColoredEvaluation.h"

#include "transposition_table.h"
#include "negamax.h"
#include "../utils/Log.h"

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
SearchResult<TURN> negamax_result_to_search_result(const NegamaxResult<TURN>& result, SearchThread* thread) {
  std::vector<Variation<TURN>> convertedPVs;
  for (const auto& pv : thread->primaryVariations_) {
    std::vector<Move> moves;
      extract_variation_from_tt(thread->position_, thread->shared_->tt, &moves, pv.first);
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
SearchResult<TURN> search(SearchThread* thread, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<TURN>, uint64_t, uint64_t)> onDepthCompleted) {
  LOG("[search] begin fen=%s threads=%u depth_limit=%u node_limit=%llu pondering=%d",
    thread->position_.fen().c_str(),
    thread->shared_->numThreads,
    thread->shared_->depthLimit,
    (unsigned long long)thread->shared_->nodeLimit,
    thread->shared_->isPondering.load() ? 1 : 0);
  thread->shared_->tt->new_search();
  auto startTime = std::chrono::high_resolution_clock::now();
  assert(thread->position_.turn_ == TURN);
  std::atomic<bool> neverStopThinking{false};
  std::vector<std::pair<std::unique_ptr<SearchThread>, std::unique_ptr<std::thread>>> otherThreads(thread->shared_->numThreads - 1);
  for (unsigned i = 0; i < otherThreads.size(); ++i) {
    otherThreads[i].first = std::make_unique<SearchThread>(
      i + 1,
      thread->position_,
      thread->shared_
    );
    otherThreads[i].second = std::make_unique<std::thread>(
      negamax<TURN, SearchType::ROOT, true>,
      otherThreads[i].first.get(),
      /*depth=*/1,
      /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
      /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
      /*plyFromRoot=*/0,
      otherThreads[i].first->root_frame(),
      &neverStopThinking
    );
  }
  LOG("[search] depth=1 starting");
  NegamaxResult<TURN> result;
  if (otherThreads.size() > 0) {
    result = negamax<TURN, SearchType::ROOT, true>(
      thread,
      1,
      /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
      /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
      /*plyFromRoot=*/0,
      thread->root_frame(),
      &neverStopThinking  // Guarantee we always search at least depth 1 before stopping.
    );
  } else {
    result = negamax<TURN, SearchType::ROOT, false>(
      thread,
      1,
      /*alpha=*/ColoredEvaluation<TURN>(kMinEval),
      /*beta=*/ColoredEvaluation<TURN>(kMaxEval),
      /*plyFromRoot=*/0,
      thread->root_frame(),
      &neverStopThinking  // Guarantee we always search at least depth 1 before stopping.
    );
  }
  LOG("[search] depth=1 joining %u worker threads", (unsigned)otherThreads.size());
  for (unsigned i = 0; i < otherThreads.size(); ++i) {
    otherThreads[i].second->join();
  }
  LOG("[search] depth=1 complete nodes=%llu", (unsigned long long)thread->nodeCount_);
  if (result.bestMove == kNullMove) {
    std::cout << "Error (1): Search did not find a move. " << thread->position_.currentState_.hash << std::endl;
    exit(1);
  }
  SearchResult<TURN> searchResult = negamax_result_to_search_result<TURN>(result, thread);
  SearchResult<TURN> lastResult = searchResult;
  if (onDepthCompleted != nullptr) {
    onDepthCompleted(1, searchResult, thread->nodeCount_, thread->qNodeCount_);
  }
  bool quitEarly = false;
  for (unsigned i = 2; i <= std::min(thread->shared_->depthLimit, kMaxSearchDepth) && !quitEarly; ++i) {
    if (stopThinking->load()) {
      LOG("[search] stopping before depth=%u (stopThinking)", i);
      break;
    }
    LOG("[search] starting depth=%u", i);
    // Experiment results:
    //  1 Win50     :     7.6    3.7  1233.5    2400    51
    //  2 Win25     :    -0.5    3.8  1198.0    2400    50
    //  3 Win100    :    -1.1    3.7  1195.0    2400    50
    //  4 Old       :    -6.0    3.7  1173.5    2400    49
    constexpr Evaluation kWindowSize = 50;
    ColoredEvaluation<TURN> alpha = (thread->shared_->multiPV == 1) ? lastResult.evaluation - kWindowSize : ColoredEvaluation<TURN>(kMinEval);
    ColoredEvaluation<TURN> beta = (thread->shared_->multiPV == 1) ? lastResult.evaluation + kWindowSize : ColoredEvaluation<TURN>(kMaxEval);
    while (true) {
      // If we're unlikely to complete this search window in time, return early. This gives
      // us more time on subsequent moves.
      if (thread->shared_->isTimeSensitive && !thread->shared_->isPondering.load()) {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTimeDuringLastSearch = endTime - startTime;
        std::chrono::duration<double> timeRemaining = thread->shared_->stopTime - endTime;
        if (elapsedTimeDuringLastSearch * 2 > timeRemaining) {
          LOG("[search] quitting early at depth=%u (time pressure)", i);
          quitEarly = true;
          break;
        }
        startTime = endTime;
      }
      for (unsigned j = 0; j < otherThreads.size(); ++j) {
        otherThreads[j].second = std::make_unique<std::thread>(
          negamax<TURN, SearchType::ROOT, true>,
          otherThreads[j].first.get(),
          /*depth=*/i,
          /*alpha=*/alpha,
          /*beta=*/beta,
          /*plyFromRoot=*/0,
          otherThreads[j].first->root_frame(),
          stopThinking
        );
      }
      if (otherThreads.size() > 0) {
        LOG("[search] depth=%u joining %u worker threads", i, (unsigned)otherThreads.size());
      }
      if (otherThreads.size() > 0) {
        result = negamax<TURN, SearchType::ROOT, true>(
          thread,
          i,
          /*alpha=*/alpha,
          /*beta=*/beta,
          /*plyFromRoot=*/0,
          thread->root_frame(),
          stopThinking
        );
      } else {
        result = negamax<TURN, SearchType::ROOT, false>(
          thread,
          i,
          /*alpha=*/alpha,
          /*beta=*/beta,
          /*plyFromRoot=*/0,
          thread->root_frame(),
          stopThinking
        );
      }
      for (unsigned j = 0; j < otherThreads.size(); ++j) {
        otherThreads[j].second->join();
      }
      if (otherThreads.size() > 0) {
        LOG("[search] depth=%u worker threads joined", i);
      }
      if (result.evaluation <= alpha) {
        alpha = ColoredEvaluation<TURN>(kMinEval);
      } else if (result.evaluation >= beta) {
        beta = ColoredEvaluation<TURN>(kMaxEval);
      } else {
        break;
      }
    }
    searchResult = negamax_result_to_search_result<TURN>(result, thread);
    // TODO: why do we need "searchResult.bestMove == kNullMove" in the condition?
    if (stopThinking->load() || quitEarly || searchResult.bestMove == kNullMove) {
      LOG("[search] returning early at depth=%u stop=%d quit_early=%d null_move=%d",
        i,
        stopThinking->load() ? 1 : 0,
        quitEarly ? 1 : 0,
        searchResult.bestMove == kNullMove ? 1 : 0);
      return lastResult;
    }
    if (searchResult.bestMove == kNullMove) {
      std::cout << "Error (2): Search did not find a move." << searchResult << " (" << thread->position_.currentState_.hash << ")" << std::endl;
      exit(1);
    }
    lastResult = searchResult;
    if (onDepthCompleted != nullptr) {
      onDepthCompleted(i, searchResult, thread->nodeCount_, thread->qNodeCount_);
    }
    if (thread->shared_->isTimeSensitive && (result.evaluation.value <= kLongestForcedMate || result.evaluation.value >= -kLongestForcedMate)) {
      // If we're in an actual game, stop searching deeper once we find a forced mate.
      LOG("[search] stopping after mate found at depth=%u", i);
      break;
    }
    LOG("[search] depth=%u complete best=%s score=%d nodes=%llu",
      i,
      searchResult.bestMove.uci().c_str(),
      searchResult.evaluation.value,
      (unsigned long long)thread->nodeCount_);
  }
  LOG("[search] finished best=%s score=%d nodes=%llu",
    searchResult.bestMove.uci().c_str(),
    searchResult.evaluation.value,
    (unsigned long long)thread->nodeCount_);
  return searchResult;
}

// Non-color-templated search function to be used by the UCI interface.
SearchResult<Color::WHITE> colorless_search(
  SearchThread* thread,
  std::atomic<bool> *stopThinking,
  std::function<void(int, SearchResult<Color::WHITE>, uint64_t, uint64_t)> onDepthCompleted
);

// Convenience function to search programmatically without needing to specify color or create a thread.
SearchResult<Color::WHITE> search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable* tt);

}  // namespace ChessEngine

#endif  // SEARCH_H
