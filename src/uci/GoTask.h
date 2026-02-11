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
#include <thread>
#include <unordered_set>

namespace ChessEngine {

struct GoCommand {
  GoCommand()
  : depthLimit(kMaxSearchDepth), nodeLimit(-1), timeLimitMs(-1),
  wtimeMs(0), btimeMs(0), wIncrementMs(0), bIncrementMs(0), movesUntilTimeControl(-1), makeBestMove(false) {}

  Position pos;

  size_t depthLimit;
  uint64_t nodeLimit;
  uint64_t timeLimitMs;
  std::unordered_set<Move> moves;

  uint64_t wtimeMs;
  uint64_t btimeMs;
  uint64_t wIncrementMs;
  uint64_t bIncrementMs;
  uint64_t movesUntilTimeControl;

  // If true, the best (found) move is made after the command finishes.
  bool makeBestMove;
};

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
  GoTask(std::deque<std::string> command) : command(command), isRunning(false), thread(nullptr) {}
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
          timeForMoveMs = goCommand.wtimeMs / goCommand.movesUntilTimeControl;
        } else {
          timeForMoveMs = goCommand.wtimeMs / 30;  // Assume 30 moves remaining if not specified.
        }
        timeForMoveMs += goCommand.wIncrementMs;
      } else {
        if (goCommand.movesUntilTimeControl != (uint64_t)-1) {
          timeForMoveMs = goCommand.btimeMs / goCommand.movesUntilTimeControl;
        } else {
          timeForMoveMs = goCommand.btimeMs / 30;  // Assume 30 moves remaining if not specified.
        }
        timeForMoveMs += goCommand.bIncrementMs;
      }
      std::cout << "Time for move: " << timeForMoveMs << " ms" <<  std::endl;
      // Use 95% of the calculated time to leave some buffer.
      goCommand.timeLimitMs = timeForMoveMs * 95 / 100;
    }

    this->baseThreadState = std::make_shared<Thread>(
      /* thread id=*/ 0,
      state->position,
      state->evaluator,
      state->multiPV,
      std::unordered_set<Move>(),
      state->tt_.get()
    );
    this->baseThreadState->depth_ = goCommand.depthLimit;
    this->baseThreadState->nodeLimit_ = goCommand.nodeLimit;
    state->stopThinking.store(false);
    if (goCommand.timeLimitMs != (uint64_t)-1) {
      this->baseThreadState->stopTime_ = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(goCommand.timeLimitMs);
      std::thread([state, timeLimitMs = goCommand.timeLimitMs]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(timeLimitMs));
        state->stopThinking.store(true);
      }).detach();
    } else {
      this->baseThreadState->stopTime_ = std::chrono::high_resolution_clock::time_point::max();
    }
    this->thread = new std::thread(GoTask::_threaded_think, this->baseThreadState.get(), state, &isRunning, isTimeSensitive);
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

  static void _threaded_think(Thread* baseThread, UciEngineState* state, bool* isRunning, bool timeSensitive) {

    // TODO: support more than one thread.
    Thread thread0 = *baseThread;

    auto startTime = std::chrono::high_resolution_clock::now();

    SearchResult<Color::WHITE> result = colorless_search(&thread0, &(state->stopThinking), [state, &thread0, &startTime](int depth, SearchResult<Color::WHITE> result) {
      auto now = std::chrono::high_resolution_clock::now();
      double secs = std::chrono::duration<double>(now - startTime).count();
      GoTask::_print_variations(depth, secs, result, state, &thread0);
    }, /*timeSensitive=*/timeSensitive);

    std::cout << "bestmove " << result.bestMove.uci() << std::endl;

    *isRunning = false;
    // Notify run-loop that it can start running a new command.
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
 private:
  static void _print_variations(int depth, double secs, SearchResult<Color::WHITE> result, UciEngineState* state, Thread* thread) {
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
  std::shared_ptr<Thread> baseThreadState;
  bool isRunning;
};

}  // namespace Nnue

#endif  // SRC_UCI_GOTASK_H