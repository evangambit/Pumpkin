#include "search.h"

namespace ChessEngine {

SearchResult<Color::WHITE> colorless_search(Thread* thread, int depth, std::function<void(int, SearchResult<Color::WHITE>)> onDepthCompleted) {
  if (thread->position_.turn_ == Color::WHITE) {
    return search<Color::WHITE>(thread, depth, onDepthCompleted);
  } else {
    if (onDepthCompleted != nullptr) {
      std::function<void(int, SearchResult<Color::BLACK>)> wrappedOnDepthCompleted =
        [onDepthCompleted](int depth, SearchResult<Color::BLACK> resultBlack) {
          SearchResult<Color::WHITE> resultWhite = -resultBlack;
          onDepthCompleted(depth, resultWhite);
        };
      return -search<Color::BLACK>(thread, depth, wrappedOnDepthCompleted);
    }
    else {
      return -search<Color::BLACK>(thread, depth, nullptr);
    }
  }
}

// Convenience function to search programmatically without needing to specify color or create a thread.
SearchResult<Color::WHITE> search(Position pos, std::shared_ptr<EvaluatorInterface> evaluator, int depth, int multiPV, TranspositionTable* tt) {
  pos.set_listener(evaluator);
  tt->new_search();
  Thread thread(0, pos, evaluator, multiPV, std::unordered_set<Move>(), tt);

  if (pos.turn_ == Color::WHITE) {
    return search<Color::WHITE>(&thread, depth, nullptr);
  } else {
    SearchResult<Color::BLACK> result = search<Color::BLACK>(&thread, depth, nullptr);
    return -result;
  }
}

}  // namespace ChessEngine
