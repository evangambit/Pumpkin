#include "search.h"

namespace ChessEngine {

SearchResult<Color::WHITE> colorless_search(
  Thread* thread,
  std::atomic<bool> *stopThinking,
  std::function<void(int, SearchResult<Color::WHITE>)> onDepthCompleted,
  bool timeSensitive
) {
  if (thread->position_.turn_ == Color::WHITE) {
    return search<Color::WHITE>(thread, stopThinking, onDepthCompleted, timeSensitive);
  } else {
    if (onDepthCompleted != nullptr) {
      std::function<void(int, SearchResult<Color::BLACK>)> wrappedOnDepthCompleted =
        [onDepthCompleted](int depth, SearchResult<Color::BLACK> resultBlack) {
          SearchResult<Color::WHITE> resultWhite = -resultBlack;
          onDepthCompleted(depth, resultWhite);
        };
      return -search<Color::BLACK>(thread, stopThinking, wrappedOnDepthCompleted, timeSensitive);
    }
    else {
      return -search<Color::BLACK>(thread, stopThinking, nullptr, timeSensitive);
    }
  }
}

// Convenience function to search programmatically without needing to specify color or create a thread.
SearchResult<Color::WHITE> search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable* tt) {
  pos.set_listener(evaluator);
  tt->new_search();
  Thread thread(0, pos, evaluator, multiPV, std::unordered_set<Move>(), tt);
  thread.depth_ = depth;
  std::atomic<bool> stopThinking {false};

  if (pos.turn_ == Color::WHITE) {
    return search<Color::WHITE>(&thread, &stopThinking, nullptr, /*timeSensitive=*/false);
  } else {
    SearchResult<Color::BLACK> result = search<Color::BLACK>(&thread, &stopThinking, nullptr, /*timeSensitive=*/false);
    return -result;
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

Evaluation increment_mate(Evaluation eval, Evaluation delta) {
  if (eval <= kLongestForcedMate) {
    return Evaluation(eval + delta);
  } else if (eval >= -kLongestForcedMate) {
    return Evaluation(eval - delta);
  } else {
    return eval;
  }
}

}  // namespace ChessEngine
