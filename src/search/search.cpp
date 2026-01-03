#include "search.h"

namespace ChessEngine {

SearchResult<Color::WHITE> colorless_search(Thread* thread, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<Color::WHITE>)> onDepthCompleted) {
  if (thread->position_.turn_ == Color::WHITE) {
    return search<Color::WHITE>(thread, stopThinking, onDepthCompleted);
  } else {
    if (onDepthCompleted != nullptr) {
      std::function<void(int, SearchResult<Color::BLACK>)> wrappedOnDepthCompleted =
        [onDepthCompleted](int depth, SearchResult<Color::BLACK> resultBlack) {
          SearchResult<Color::WHITE> resultWhite = -resultBlack;
          onDepthCompleted(depth, resultWhite);
        };
      return -search<Color::BLACK>(thread, stopThinking, wrappedOnDepthCompleted);
    }
    else {
      return -search<Color::BLACK>(thread, stopThinking, nullptr);
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
    return search<Color::WHITE>(&thread, &stopThinking, nullptr);
  } else {
    SearchResult<Color::BLACK> result = search<Color::BLACK>(&thread, &stopThinking, nullptr);
    return -result;
  }
}

void extract_variation_from_tt(const Position& pos, TranspositionTable* tt, std::vector<Move>* movesOut, Move startMove) {
  Position position = pos;
  Move move = startMove;
  while (move != kNullMove) {
    ez_make_move(&position, move);
    movesOut->push_back(move);
    TTEntry entry;
    if (!tt->probe(position.currentState_.hash, entry)) {
      break;
    }
    move = entry.bestMove;
  }
}

}  // namespace ChessEngine
