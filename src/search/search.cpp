#include "search.h"
#include "negamax.h"

namespace ChessEngine {

SearchResult<Color::WHITE> colorless_search(
  SearchThread* thread,
  std::atomic<bool> *stopThinking,
  std::function<void(int, SearchResult<Color::WHITE>, uint64_t, uint64_t)> onDepthCompleted
) {
  if (thread->position_.turn_ == Color::WHITE) {
    return search<Color::WHITE>(thread, stopThinking, onDepthCompleted);
  } else {
    if (onDepthCompleted != nullptr) {
      std::function<void(int, SearchResult<Color::BLACK>, uint64_t, uint64_t)> wrappedOnDepthCompleted =
        [onDepthCompleted](int depth, SearchResult<Color::BLACK> resultBlack, uint64_t nodeCount, uint64_t qNodeCount) {
          SearchResult<Color::WHITE> resultWhite = -resultBlack;
          onDepthCompleted(depth, resultWhite, nodeCount, qNodeCount);
        };
      return -search<Color::BLACK>(thread, stopThinking, wrappedOnDepthCompleted);
    }
    else {
      return -search<Color::BLACK>(thread, stopThinking, nullptr);
    }
  }
}

void extract_variation_from_tt(const Position& pos, TranspositionTable* tt, std::vector<Move>* movesOut, Move startMove) {
  Position position = pos;
  Move move = startMove;
  assert(move != kNullMove);
  std::unordered_set<uint64_t> visitedHashes;
  visitedHashes.insert(position.currentState_.hash);
  while (move != kNullMove) {
    ez_make_move(&position, move);
    movesOut->push_back(move);
    if (movesOut->size() >= 10) {
      break;
    }
    if (visitedHashes.count(position.currentState_.hash) > 0) {
      break;
    }
    TTEntry entry;
    if (!tt->probe(position.currentState_.hash, entry)) {
      break;
    }
    move = entry.bestMove;
  }
}

SearchResult<Color::WHITE> search(
  Position pos,
  std::shared_ptr<EvaluatorInterface> evaluator,
  int depth,
  int multiPV,
  TranspositionTable* tt
) {
  pos.set_listener(evaluator);

  GoCommand command;
  command.depthLimit = depth;
  command.nodeLimit = static_cast<uint64_t>(-1);
  command.timeLimitMs = static_cast<uint64_t>(-1);

  const auto stopTime =
    std::chrono::high_resolution_clock::now() + std::chrono::hours(24);
  auto shared = std::make_shared<SharedSearchThreadState>(
    command,
    static_cast<unsigned>(multiPV),
    /*numThreads=*/1,
    /*isTimeSensitive=*/false,
    stopTime,
    tt
  );

  SearchThread thread(0, pos, shared);
  std::atomic<bool> stopFlag(false);
  if (pos.turn_ == Color::WHITE) {
    return search<Color::WHITE>(&thread, &stopFlag, nullptr);
  }
  return -search<Color::BLACK>(&thread, &stopFlag, nullptr);
}

}  // namespace ChessEngine
