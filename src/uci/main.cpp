#include "../search/search.h"
#include "../game/Position.h"
#include "../game/movegen/movegen.h"
#include "../game/Utils.h"
#include "../utils/StringUtils.h"
#include "../utils/Log.h"
#include "GoTask.h"
#include "SelfPlayTask.h"
#include "Task.h"
#include "TrivialTasks.h"
#include "SetOptionTask.h"
#include "PositionTask.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

using namespace ChessEngine;

static void finish_current_uci_command(UciEngineState *state) {
  if (state->currentCommandActive) {
    LOG("[uci] end: %s", state->currentCommandText.c_str());
    state->currentCommandActive = false;
    state->currentCommandText.clear();
  }
}

static void enqueue_uci_task(UciEngineState *state, const std::string& command, std::shared_ptr<Task> task) {
  state->taskQueue.push_back(QueuedTask{command, task});
}

class IsReadyTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << "readyok" << std::endl;
  }
};

bool wait_for_task(UciEngineState *state) {
  state->taskQueueLock.lock();
  if (state->taskQueue.size() > 0) {
    state->taskQueueLock.unlock();
    return true;
  }
  if (state->shuttingDown.load()) {
    state->taskQueueLock.unlock();
    return false;
  }
  state->taskQueueLock.unlock();
  LOG("[eventRunner] waiting for task (queue empty)");
  while (true) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.wait(lock);  // Wait for data
    state->taskQueueLock.lock();
    if (state->taskQueue.size() > 0) {
      LOG("[eventRunner] woke up, queue size=%zu", state->taskQueue.size());
      state->taskQueueLock.unlock();
      return true;
    }
    if (state->shuttingDown.load()) {
      state->taskQueueLock.unlock();
      return false;
    }
    state->taskQueueLock.unlock();
  }
}

struct UciEngine {
  UciEngineState state;

  UciEngine() {}
  static void print_preamble(UciEngineState *state) {
    std::cout << "id name " << state->name << std::endl;
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
  void start(std::istream& cin, const std::vector<std::string>& commands) {
    UciEngineState *state = &this->state;

    if (commands.size() == 0) {
      print_preamble(state);
      LOG("[uci] end: uci");
    }

    for (std::string command : commands) {
      if(command.find_first_not_of(' ') == std::string::npos) {
        continue;
      }
      this->handle_uci_command(state, &command);
    }

    std::thread eventRunner([state]() {
      LOG("[eventRunner] started");
      while (true) {
        if (!wait_for_task(state)) {
          LOG("[eventRunner] exiting (shutdown)");
          return;
        }

        // Wait until not busy.
        state->taskQueueLock.lock();
        while (state->currentTask != nullptr && state->currentTask->is_running()) {
          LOG("[eventRunner] blocked waiting for current task to finish (running=%d queue=%zu)",
            state->currentTask->is_running() ? 1 : 0,
            state->taskQueue.size());
          state->taskQueueLock.unlock();
          std::unique_lock<std::mutex> lock(state->mutex);
          state->condVar.wait(lock);  // Wait for data
          state->taskQueueLock.lock();
        }
        finish_current_uci_command(state);

        if (state->taskQueue.size() == 0) {
          state->taskQueueLock.unlock();
          throw std::runtime_error("No task to enque");
        }

        QueuedTask queued = state->taskQueue.front();
        state->taskQueue.pop_front();
        state->currentTask = queued.task;
        state->currentCommandText = queued.command;
        state->taskQueueLock.unlock();

        LOG("[uci] start: %s", state->currentCommandText.c_str());
        state->currentCommandActive = true;
        state->currentTask->start(state);
        if (!state->currentTask->is_running()) {
          finish_current_uci_command(state);
        }
      }
    });
    while (true) {
      if (cin.eof()) {
        break;
      }
      std::string line;
      if (!getline(cin, line)) {
        break;
      }
      if (line == "quit") {
        LOG("[uci] received: quit");
        LOG("[uci] start: quit");
        state->shuttingDown.store(true);
        std::unique_lock<std::mutex> lock(this->state.mutex);
        this->state.condVar.notify_all();
        exit(0);
        break;
      }

      // Skip empty lines.
      if(line.find_first_not_of(' ') == std::string::npos) {
        continue;
      }

      this->handle_uci_command(state, &line);
      std::unique_lock<std::mutex> lock(this->state.mutex);
      this->state.condVar.notify_one();
    }
    state->shuttingDown.store(true);
    {
      std::unique_lock<std::mutex> lock(this->state.mutex);
      this->state.condVar.notify_all();
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

    if (parts.empty()) {
      return;
    }

    LOG("[uci] received: %s", command->c_str());

    state->taskQueueLock.lock();
    if (parts[0] == "position" || parts[0] == "p") {
      enqueue_uci_task(state, *command, std::make_shared<PositionTask>(parts));
    } else if (parts[0] == "go") {
      enqueue_uci_task(state, *command, std::make_shared<GoTask>(parts));
    } else if (parts[0] == "setoption" || parts[0] == "so") {
      enqueue_uci_task(state, *command, std::make_shared<SetOptionTask>(parts));
    } else if (parts[0] == "ucinewgame") {
      enqueue_uci_task(state, *command, std::make_shared<NewGameTask>());
    } else if (parts[0] == "stop") {
      // This runs immediately.
      LOG("[uci] start: %s", command->c_str());
      StopTask task;
      task.start(state);
      LOG("[uci] end: %s", command->c_str());
    } else if (parts[0] == "printoptions") {
      enqueue_uci_task(state, *command, std::make_shared<PrintOptionsTask>());
    } else if (parts[0] == "isready") {
      enqueue_uci_task(state, *command, std::make_shared<IsReadyTask>());
    } else if (parts[0] == "move" || parts[0] == "m") {
      enqueue_uci_task(state, *command, std::make_shared<MoveTask>(parts));
    } else if (parts[0] == "hash") {
      enqueue_uci_task(state, *command, std::make_shared<HashTask>());
    } else if (parts[0] == "lazyquit") {
      enqueue_uci_task(state, *command, std::make_shared<QuitTask>());
    } else if (parts[0] == "printfen") {
      enqueue_uci_task(state, *command, std::make_shared<PrintFenTask>());
    } else if (parts[0] == "silence") {
      enqueue_uci_task(state, *command, std::make_shared<SilenceTask>(parts));
    #ifdef PRINT_DEBUG
    } else if (parts[0] == "printdebug") {
      enqueue_uci_task(state, *command, std::make_shared<PrintDebugTask>(parts));
    #endif
    } else if (parts[0] == "uci") {
      // It is convenient to pretend this is successful so we can set stuff up before cutechess-cli does its thing.
      LOG("[uci] start: %s", command->c_str());
      print_preamble(state);
      LOG("[uci] end: %s", command->c_str());
    } else if (parts[0] == "ponderhit") {
      LOG("[uci] start: %s", command->c_str());
      if (state->sharedSearchThreadState) {
        state->sharedSearchThreadState->ponderHit();
      }
      LOG("[uci] end: %s", command->c_str());
    } else if (parts[0] == "probe") {
      // For probing the TT from UCI.
      enqueue_uci_task(state, *command, std::make_shared<ProbeTask>(parts));
    } else if (parts[0] == "debug") {
      LOG("[uci] start: %s", command->c_str());
      // Todo
      LOG("[uci] end: %s", command->c_str());
    } else if (parts[0] == "evaluator") {
      enqueue_uci_task(state, *command, std::make_shared<SetEvaluatorTask>(parts));
    } else if (parts[0] == "selfplay") {
      enqueue_uci_task(state, *command, std::make_shared<SelfPlayTask>(parts));
    } else if (parts[0] == "eval") {
      enqueue_uci_task(state, *command, std::make_shared<EvalTask>(parts));
    } else if (parts[0] == "fenerror") {
      enqueue_uci_task(state, *command, std::make_shared<FenErrorTask>(parts));
    } else if (parts[0] == "nnueevaldebug") {
      enqueue_uci_task(state, *command, std::make_shared<NnueEvalDebugTask>(parts));
    } else if (parts[0] == "byhandevaldebug") {
      enqueue_uci_task(state, *command, std::make_shared<ByHandEvalDebugTask>(parts));
    } else if (parts[0] == "increment") {
      enqueue_uci_task(state, *command, std::make_shared<IncrementWeightTask>(parts));
    } else if (parts[0] == "increment_search") {
      enqueue_uci_task(state, *command, std::make_shared<IncrementSearchHyperParamTask>(parts));
    } else if (parts[0] == "dumpweights") {
      enqueue_uci_task(state, *command, std::make_shared<DumpWeightsTask>(parts));
    } else if (parts[0] == "analyzefens") {
      enqueue_uci_task(state, *command, std::make_shared<AnalyzeFens>(parts));
    } else {
      enqueue_uci_task(state, *command, std::make_shared<UnrecognizedCommandTask>(parts));
    }
    state->taskQueueLock.unlock();

    std::unique_lock<std::mutex> lock(state->mutex);
    state->condVar.notify_one();
  }
};

int main(int argc, char *argv[]) {
  std::string arg0 = argv[0];
  LOG_INIT(arg0 + "-pumpkin.log");
  LOG("Engine starting");

  std::string name = std::string("Pumpkin 0.0 (") + argv[0] + ")";
  std::cout << name << std::endl;

  std::vector<std::string> commands;
  for (int i = 1; i < argc; ++i) {
    commands.push_back(argv[i]);
  }

  // Wait for "uci" command.
  if (commands.size() == 0) {
    while (true) {
      std::string line;
      if (!getline(std::cin, line)) {
        return 0;
      }
      if (line == "uci") {
        LOG("[uci] received: uci");
        LOG("[uci] start: uci");
        break;
      } else {
        LOG("[uci] received: %s", line.c_str());
        std::cout << "Unrecognized command " << repr(line) << std::endl;
      }
    }
  }

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  UciEngine engine;
  engine.state.name = name;
  engine.start(std::cin, commands);

  LOG("Engine shutting down");
  LOG_SHUTDOWN();
}
