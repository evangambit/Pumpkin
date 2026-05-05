#ifndef SRC_UCI_GOTASK_H
#define SRC_UCI_GOTASK_H

#include "../search/search.h"
#include "../search/negamax.h"
#include "../game/Position.h"
#include "../game/movegen/movegen.h"
#include "../game/Utils.h"
#include "Task.h"
#include "TrivialTasks.h"
#include "SetOptionTask.h"
#include "PositionTask.h"
#include "../StringUtils.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace ChessEngine {

GoCommand make_go_command(std::deque<std::string> *command, Position *pos) {
  GoCommand goCommand;

  goCommand.pos = *pos;

  std::unordered_set<std::string> uciMoves;
  std::string lastCommand = "";
  while (command->size() > 0) {
    std::string part = command->front();
    command->pop_front();

    if (part == "depth"
      || part == "nodes"
      || part == "movetime"
      || part == "wtime"
      || part == "btime"
      || part == "winc"
      || part == "binc"
      || part == "movestogo"
      || part == "searchmoves"
      ) {
      lastCommand = part;
    } else if (part == "mm") {
      goCommand.makeBestMove = true;
    } else if (lastCommand == "depth") {
      goCommand.depthLimit = std::min(stoull(part), (unsigned long long)kMaxSearchDepth);
    } else if (lastCommand == "nodes") {
      goCommand.nodeLimit = stoull(part);
    } else if (lastCommand == "movetime") {
      goCommand.timeLimitMs = stoull(part);
    } else if (lastCommand == "wtime") {
      goCommand.wtimeMs = stoull(part);
    } else if (lastCommand == "btime") {
      goCommand.btimeMs = stoull(part);
    } else if (lastCommand == "winc") {
      goCommand.wIncrementMs = stoull(part);
    } else if (lastCommand == "binc") {
      goCommand.bIncrementMs = stoull(part);
    } else if (lastCommand == "movestogo") {
      goCommand.movesUntilTimeControl = stoull(part);
    } else if (lastCommand == "searchmoves") {
      uciMoves.insert(part);
    } else {
      lastCommand = part;
    }
  }

  std::unordered_map<std::string, Move> legalMoves;
  {
    ExtMove moves[kMaxNumMoves];
    ExtMove* end;
    if (goCommand.pos.turn_ == Color::BLACK) {
      end = compute_legal_moves<Color::BLACK>(&goCommand.pos, &(moves[0]));
    } else {
      end = compute_legal_moves<Color::WHITE>(&goCommand.pos, &(moves[0]));
    }
    for (ExtMove* move = moves; move != end; ++move) {
      legalMoves.insert({move->move.uci(), move->move});
    }
  }

  // Remove invalid moves.
  for (const auto& move : uciMoves) {
    if (legalMoves.contains(move)) {
      goCommand.moves.insert(legalMoves[move]);
    }
  }

  return goCommand;
}


class GoTask : public Task {
 public:
  GoTask(std::deque<std::string> command) : command(command), thread(nullptr), isRunning(false) {}
  void start(UciEngineState *state) override {
    assert(!isRunning);
    isRunning = true;
    assert(command.at(0) == "go");
    command.pop_front();

    GoCommand goCommand = make_go_command(&command, &state->position);

    bool isTimeSensitive = false;
    if (goCommand.wtimeMs != 0 || goCommand.btimeMs != 0) {
      // We're in a timed game. Convert to a time limit.
      isTimeSensitive = true;
      uint64_t timeForMoveMs;
      if (state->position.turn_ == Color::WHITE) {
        if (goCommand.movesUntilTimeControl != (uint64_t)-1) {
          timeForMoveMs = std::max<int64_t>(0, (int64_t(goCommand.wtimeMs) - int64_t(state->moveOverheadMs)) / goCommand.movesUntilTimeControl);
        } else {
          timeForMoveMs = std::max<int64_t>(0, (int64_t(goCommand.wtimeMs) - int64_t(state->moveOverheadMs)) / 30);
        }
        timeForMoveMs += goCommand.wIncrementMs;
      } else {
        if (goCommand.movesUntilTimeControl != (uint64_t)-1) {
          timeForMoveMs = std::max<int64_t>(0, (int64_t(goCommand.btimeMs) - int64_t(state->moveOverheadMs)) / goCommand.movesUntilTimeControl);
        } else {
          timeForMoveMs = std::max<int64_t>(0, (int64_t(goCommand.btimeMs) - int64_t(state->moveOverheadMs)) / 30);
        }
        timeForMoveMs += goCommand.bIncrementMs;
      }
      goCommand.timeLimitMs = timeForMoveMs;
    }

    auto stopTime = std::chrono::high_resolution_clock::time_point::max();
    if (goCommand.timeLimitMs != (uint64_t)-1) {
      stopTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(goCommand.timeLimitMs);
    }
    if (state->sharedSearchThreadState.get() != nullptr) {
      std::cerr << "Error: Already running a search!" << std::endl;
      exit(1);
    }
    state->sharedSearchThreadState = std::make_shared<SharedSearchThreadState>(goCommand, state->multiPV, isTimeSensitive, stopTime, state->tt_.get());
    this->baseThreadState = std::make_shared<SearchThread>(
      /* thread id=*/ 0,
      state->position,
      state->sharedSearchThreadState,
      goCommand
    );
    state->stopThinking = std::make_shared<std::atomic<bool>>(false);
    auto currentStopThinking = state->stopThinking;
    this->thread = new std::thread(GoTask::_threaded_think, this->baseThreadState.get(), state, currentStopThinking, &isRunning);
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

  static void _threaded_think(SearchThread* baseThread, UciEngineState* state, std::shared_ptr<std::atomic<bool>> stopThinking, bool* isRunning) {

    // TODO: support more than one thread.
    SearchThread thread0 = *baseThread;

    auto startTime = std::chrono::high_resolution_clock::now();

    SearchResult<Color::WHITE> result = colorless_search(&thread0, stopThinking.get(), [state, &thread0, &startTime](int depth, SearchResult<Color::WHITE> result) {
      auto now = std::chrono::high_resolution_clock::now();
      double secs = std::chrono::duration<double>(now - startTime).count();
      GoTask::_print_variations(depth, secs, result, state, &thread0);
    });

    std::cout << "bestmove " << result.bestMove.uci() << std::endl;

    *isRunning = false;
    state->sharedSearchThreadState = nullptr;
    // Notify run-loop that it can start running a new command.
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
 private:
  static void _print_variations(int depth, double secs, SearchResult<Color::WHITE> result, UciEngineState* state, SearchThread* thread) {
    const size_t multiPV = state->multiPV;
    const uint64_t timeMs = secs * 1000;
    if (result.primaryVariations.size() == 0) {
      if (colorless_is_stalemate(&state->position)) {
        std::cout << "info depth 0 score cp 0" << std::endl;
        return;
      } else {
        throw std::runtime_error("todo");
      }
    }
    for (size_t i = 0; i < std::min(multiPV, result.primaryVariations.size()); ++i) {
      Variation<Color::WHITE> variation = result.primaryVariations[i];

      Evaluation eval = variation.evaluation.value;
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
      std::cout << " qnodes " << thread->qNodeCount_;
      std::cout << " nps " << uint64_t(double(thread->nodeCount_) / secs);
      std::cout << " time " << timeMs;
      std::cout << " pv";
      for (const auto& move : variation.moves) {
        std::cout << " " << move.uci();
      }
      std::cout << std::endl;
    }
  }
  std::deque<std::string> command;
  std::thread *thread;
  std::shared_ptr<SearchThread> baseThreadState;
  bool isRunning;
};

}  // namespace ChessEngine

#endif  // SRC_UCI_GOTASK_H