#ifndef SRC_UCI_GOTASK_H
#define SRC_UCI_GOTASK_H

#include "../search/search.h"
#include "../game/Position.h"
#include "../game/movegen/movegen.h"
#include "../game/utils.h"
#include "Task.h"
#include "TrivialTasks.h"
#include "SetOptionTask.h"
#include "PositionTask.h"
#include "../string_utils.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace ChessEngine {

class GoTask : public Task {
 public:
  GoTask(std::deque<std::string> command) : command(command), isRunning(false), thread(nullptr) {}
  void start(UciEngineState *state) override {
    assert(!isRunning);
    isRunning = true;
    assert(command.at(0) == "go");
    command.pop_front();

    this->baseThreadState = std::make_shared<Thread>(
      /* thread id=*/ 0,
      state->position,
      state->evaluator,
      state->multiPV,
      std::unordered_set<Move>(),
      state->tt_.get()
    );
    this->thread = new std::thread(GoTask::_threaded_think, this->baseThreadState.get(), state, &isRunning);
  }

  bool is_running() override {
    return isRunning;
  }

  ~GoTask() {
    assert(!isRunning);
    assert(this->thread != nullptr);
    this->thread->join();
    delete this->thread;
  }

  static void _threaded_think(Thread* baseThread, UciEngineState* state, bool* isRunning) {

    // TODO: support more than one thread.
    Thread thread0 = baseThread->clone();
    thread0.depth_ = 4;

    auto startTime = std::chrono::high_resolution_clock::now();

    SearchResult<Color::WHITE> result = colorless_search(&thread0, &(state->stopThinking), [state, &thread0, &startTime](int depth, SearchResult<Color::WHITE> result) {
      auto now = std::chrono::high_resolution_clock::now();
      double secs = std::chrono::duration<double>(now - startTime).count();
      GoTask::_print_variations(depth, secs, result, state, &thread0);
    });
    *isRunning = false;
    // Notify run-loop that it can start running a new command.
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
 private:
  static void _print_variations(int depth, double secs, SearchResult<Color::WHITE> result, UciEngineState* state, Thread* thread) {
    const size_t multiPV = state->multiPV;
    const uint64_t timeMs = secs * 1000;
    std::cout << depth << " depth completed in " << timeMs << " ms, " << thread->nodeCount_ << " nodes searched." << std::endl;
    if (result.primaryVariations.size() == 0) {
      if (colorless_is_stalemate(&state->position)) {
        std::cout << "info depth 0 score cp 0" << std::endl;
        return;
      } else {
        throw std::runtime_error("todo");
      }
    }
    for (size_t i = 0; i < std::min(multiPV, result.primaryVariations.size()); ++i) {
      std::pair<Evaluation, std::vector<Move>> variation = std::make_pair(result.primaryVariations[i].second.value, std::vector<Move>({result.primaryVariations[i].first}));

      Evaluation eval = variation.first;
      if (state->position.turn_ == Color::BLACK) {
        // Score should be from mover's perspective, not white's.
        eval *= -1;
      }

      std::cout << "info depth " << depth;
      std::cout << " multipv " << (i + 1);
      if (eval <= kLongestForcedMate) {
        std::cout << " score mate " << -(eval - kCheckmate + 1) / 2;
      } else if (eval >= -kLongestForcedMate) {
        std::cout << " score mate " << -(eval + kCheckmate - 1) / 2;
      } else {
        std::cout << " score cp " << eval;
      }
      std::cout << " nodes " << thread->nodeCount_;
      std::cout << " nps " << uint64_t(double(thread->nodeCount_) / secs);
      std::cout << " time " << timeMs;
      std::cout << " pv";
      for (const auto& move : variation.second) {
        std::cout << " " << move.uci();
      }
      std::cout << std::endl;
    }
  }
  std::deque<std::string> command;
  std::thread *thread;
  std::shared_ptr<Thread> baseThreadState;
  bool isRunning;
};

}  // namespace ChessEngine

#endif  // SRC_UCI_GOTASK_H