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

}  // namespace ChessEngine
