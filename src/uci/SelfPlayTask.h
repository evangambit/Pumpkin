#ifndef SRC_UCI_SELFPLAYTASK_H
#define SRC_UCI_SELFPLAYTASK_H

#include "../search/search.h"
#include "../game/Position.h"
#include "../game/movegen/movegen.h"
#include "../game/utils.h"
#include "Task.h"

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace ChessEngine {

class SelfPlayTask : public Task {
 public:
  SelfPlayTask() : isRunning(false), thread(nullptr) {}

  void start(UciEngineState *state) override {
    std::cout << "Starting self-play task" << std::endl;
    assert(!isRunning);
    isRunning = true;
    this->thread = new std::thread(SelfPlayTask::_threaded_selfplay, state, &isRunning);
  }

  bool is_running() override {
    return isRunning;
  }

  ~SelfPlayTask() {
    assert(!isRunning);
    assert(this->thread != nullptr);
    this->thread->join();
    delete this->thread;
  }

  static void _threaded_selfplay(UciEngineState* state, bool* isRunning) {
    std::cout << "Entering self-play loop with evaluator " << state->evaluator->to_string() << std::endl;
    constexpr uint64_t kNodeLimit = 10'000'000;
    std::unordered_map<uint64_t, int> positionCounts;

    while (true) {
      // Check for threefold repetition
      uint64_t currentHash = state->position.currentState_.hash;
      positionCounts[currentHash]++;
      if (positionCounts[currentHash] >= 3) {
        std::cout << "result 1/2-1/2 {Threefold repetition}" << std::endl;
        break;
      }

      // Check for legal moves
      ExtMove moves[kMaxNumMoves];
      ExtMove* end;
      if (state->position.turn_ == Color::BLACK) {
        end = compute_legal_moves<Color::BLACK>(&state->position, &(moves[0]));
      } else {
        end = compute_legal_moves<Color::WHITE>(&state->position, &(moves[0]));
      }

      bool hasLegalMoves = (moves != end);

      if (!hasLegalMoves) {
        // No legal moves - checkmate or stalemate
        if (colorless_is_stalemate(&state->position)) {
          std::cout << "result 1/2-1/2 {Stalemate}" << std::endl;
        } else {
          // Checkmate - side to move lost
          if (state->position.turn_ == Color::WHITE) {
            std::cout << "result 0-1 {Black wins by checkmate}" << std::endl;
          } else {
            std::cout << "result 1-0 {White wins by checkmate}" << std::endl;
          }
        }
        break;
      }

      // Check for insufficient material
      if (state->position.is_material_draw()) {
        std::cout << "result 1/2-1/2 {Insufficient material}" << std::endl << std::flush;
        break;
      }

      // Create thread state for search
      Thread searchThread(
        /* thread id=*/ 0,
        state->position,
        state->evaluator,
        /* multiPV=*/ 1,
        std::unordered_set<Move>(),
        state->tt_.get()
      );
      searchThread.depth_ = kMaxSearchDepth;
      searchThread.nodeLimit_ = kNodeLimit;
      searchThread.stopTime_ = std::chrono::high_resolution_clock::time_point::max();
      if (searchThread.position_.boardListener_ != state->evaluator) {
        std::cerr << "Error: BoardListener of position does not match evaluator." << std::endl;
        exit(1);
      }

      // Search
      std::atomic<bool> neverStop{false};
      SearchResult<Color::WHITE> result = colorless_search(
        &searchThread,
        &neverStop,
        [](int depth, SearchResult<Color::WHITE> result) {
          std::cout << "info depth " << depth << " score cp " << result.evaluation.value << " move " << result.bestMove.uci() << std::endl;
        },
        /*timeSensitive=*/false
      );

      if (result.bestMove == kNullMove) {
        std::cerr << "SelfPlayError: Search did not find a move." << std::endl;
        exit(1);
      }

      // Make the move
      std::cout << "move " << result.bestMove.uci() << " score " << result.evaluation << " position fen " << state->position.fen() << std::endl;
      ez_make_move(&state->position, result.bestMove);
    }

    for (auto& move : state->position.history_) {
      std::cout << move.move.uci() << " ";
    }
    std::cout << std::endl;

    *isRunning = false;
    // Notify run-loop that it can start running a new command.
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }

 private:
  std::thread *thread;
  bool isRunning;
};

}  // namespace ChessEngine

#endif  // SRC_UCI_SELFPLAYTASK_H
