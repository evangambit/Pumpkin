#include "search/search.h"
#include "game/Position.h"
#include "game/movegen/movegen.h"
#include "game/utils.h"
#include "uci/Task.h"
#include "uci/TrivialTasks.h"
#include "uci/SetOptionTask.h"
#include "uci/PositionTask.h"
#include "string_utils.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

using namespace ChessEngine;

class IsReadyTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << "readyok" << std::endl;
  }
};

bool colorless_is_stalemate(Position *pos) {
  if (pos->turn_ == Color::WHITE) {
    return is_stalemate<Color::WHITE>(pos);
  } else {
    return is_stalemate<Color::BLACK>(pos);
  }
}

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

    // SearchResult<Color::WHITE> colorless_search(Thread* thread, int depth, std::atomic<bool> *stopThinking, std::function<void(int, SearchResult<Color::WHITE>)> onDepthCompleted);

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

void wait_for_task(UciEngineState *state) {
  state->taskQueueLock.lock();
  if (state->taskQueue.size() > 0) {
    state->taskQueueLock.unlock();
    return;
  }
  state->taskQueueLock.unlock();
  while (true) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.wait(lock);  // Wait for data
    state->taskQueueLock.lock();
    if (state->taskQueue.size() > 0) {
      state->taskQueueLock.unlock();
      return;
    }
    state->taskQueueLock.unlock();
  }
}

struct UciEngine {
  UciEngineState state;

  UciEngine() {
    this->state.position = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }
  void start(std::istream& cin, const std::vector<std::string>& commands) {
    UciEngineState *state = &this->state;

    if (commands.size() == 0) {
      std::cout << "id name Pumpkin 0.0" << std::endl;
      std::cout << "id author Morgan Redding" << std::endl << std::endl;

      // Garbage boiler plate to make the GUI happy.
      std::cout << "option name Threads type spin default 1 min 1 max 1024" << std::endl;
      std::cout << "option name Hash type spin default 16 min 1 max 33554432" << std::endl;
      std::cout << "option name Clear Hash type button" << std::endl;
      std::cout << "option name Ponder type check default false" << std::endl;
      std::cout << "option name MultiPV type spin default 1 min 1 max 500" << std::endl;
      std::cout << "option name Skill Level type spin default 20 min 0 max 20" << std::endl;
      std::cout << "option name Move Overhead type spin default 10 min 0 max 5000" << std::endl;
      std::cout << "option name Slow Mover type spin default 100 min 10 max 1000" << std::endl;
      std::cout << "option name nodestime type spin default 0 min 0 max 10000" << std::endl;
      std::cout << "option name UCI_Chess960 type check default false" << std::endl;
      std::cout << "option name UCI_AnalyseMode type check default false" << std::endl;
      std::cout << "option name UCI_LimitStrength type check default false" << std::endl;
      std::cout << "option name UCI_Elo type spin default 1320 min 1320 max 3190" << std::endl;
      std::cout << "option name UCI_ShowWDL type check default false" << std::endl;
      std::cout << "option name SyzygyPath type string default <empty>" << std::endl;
      std::cout << "option name SyzygyProbeDepth type spin default 1 min 1 max 100" << std::endl;
      std::cout << "option name Syzygy50MoveRule type check default true" << std::endl;
      std::cout << "option name SyzygyProbeLimit type spin default 7 min 0 max 7" << std::endl;

      std::cout << "uciok" << std::endl;
    }

    for (std::string command : commands) {
      if(command.find_first_not_of(' ') == std::string::npos) {
        continue;
      }
      this->handle_uci_command(state, &command);
    }

    std::thread eventRunner([state]() {
      while (true) {
        wait_for_task(state);

        // Wait until not busy.
        state->taskQueueLock.lock();
        while (state->currentTask != nullptr && state->currentTask->is_running()) {
          state->taskQueueLock.unlock();
          std::unique_lock<std::mutex> lock(state->mutex);
          state->condVar.wait(lock);  // Wait for data
          state->taskQueueLock.lock();
        }

        if (state->taskQueue.size() == 0) {
          throw std::runtime_error("No task to enque");
        }

        state->currentTask = state->taskQueue.front();
        state->taskQueue.pop_front();
        state->currentTask->start(state);
        state->taskQueueLock.unlock();
      }
    });
    while (true) {
      if (std::cin.eof()) {
        break;
      }
      std::string line;
      getline(std::cin, line);
      if (line == "quit") {
        exit(0);
        break;
      }

      // Skip empty lines.
      if(line.find_first_not_of(' ') == std::string::npos) {
        continue;
      }

      this->handle_uci_command(state, &line);

      // Notify run-loop that there may be a new command.
      std::unique_lock<std::mutex> lock(this->state.mutex);
      this->state.condVar.notify_one();
    }
    eventRunner.join();
  }
  static void handle_uci_command(UciEngineState *state, std::string *command) {
    remove_excess_whitespace(command);
    std::vector<std::string> rawParts = split(*command, ' ');

    std::deque<std::string> parts;
    for (const auto& part : rawParts) {
      if (part.size() > 0) {
        parts.push_back(part);
      }
    }

    state->taskQueueLock.lock();
    if (parts[0] == "position" || parts[0] == "p") {
      state->taskQueue.push_back(std::make_shared<PositionTask>(parts));
    } else if (parts[0] == "go") {
      state->taskQueue.push_back(std::make_shared<GoTask>(parts));
    } else if (parts[0] == "setoption" || parts[0] == "so") {
      state->taskQueue.push_back(std::make_shared<SetOptionTask>(parts));
    } else if (parts[0] == "ucinewgame") {
      state->taskQueue.push_back(std::make_shared<NewGameTask>());
    } else if (parts[0] == "stop") {
      // This runs immediately.
      StopTask task;
      task.start(state);
    } else if (parts[0] == "printoptions") {
      state->taskQueue.push_back(std::make_shared<PrintOptionsTask>());
    } else if (parts[0] == "isready") {
      state->taskQueue.push_back(std::make_shared<IsReadyTask>());
    } else if (parts[0] == "move" || parts[0] == "m") {
      state->taskQueue.push_back(std::make_shared<HashTask>());
    } else if (parts[0] == "lazyquit") {
      state->taskQueue.push_back(std::make_shared<QuitTask>());
    } else if (parts[0] == "printfen") {
      state->taskQueue.push_back(std::make_shared<PrintFenTask>());
    } else if (parts[0] == "silence") {
      state->taskQueue.push_back(std::make_shared<SilenceTask>(parts));
    #ifdef PRINT_DEBUG
    } else if (parts[0] == "printdebug") {
      state->taskQueue.push_back(std::make_shared<PrintDebugTask>(parts));
    #endif
    } else if (parts[0] == "ponderhit") {
      // Ignore. (TODO: handle pondering better).
    } else {
      state->taskQueue.push_back(std::make_shared<UnrecognizedCommandTask>(parts));
    }
    state->taskQueueLock.unlock();

    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
};

int main(int argc, char *argv[]) {
  std::cout << "Pumpkin 0.0" << std::endl;

  std::vector<std::string> commands;
  for (int i = 1; i < argc; ++i) {
    commands.push_back(argv[i]);
  }

  // Wait for "uci" command.
  if (commands.size() == 0) {
    while (true) {
      std::string line;
      getline(std::cin, line);
      if (line == "uci") {
        break;
      } else {
        std::cout << "Unrecognized command " << repr(line) << std::endl;
      }
    }
  }

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  UciEngine engine;
  engine.start(std::cin, commands);
}
