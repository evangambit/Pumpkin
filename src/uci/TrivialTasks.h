#ifndef PUMPKIN_UCI_TRIVIALTASKS_H
#define PUMPKIN_UCI_TRIVIALTASKS_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace ChessEngine {

class UnrecognizedCommandTask : public Task {
 public:
  UnrecognizedCommandTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    std::cout << "Unrecognized command \"" << join(command, " ") << "\"" << std::endl;
  }
 private:
  std::deque<std::string> command;
};

class HashTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << state->position.currentState_.hash << std::endl;
  }
};

class PrintFenTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << state->position.fen() << std::endl;
  }
};

class SilenceTask : public Task {
 public:
  SilenceTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.at(1) == "1") {
      std::cout.setstate(std::ios::failbit);
    } else {
      std::cout.clear();
    }
  }
 private:
  std::deque<std::string> command;
};


class PrintOptionsTask : public Task {
 public:
  void start(UciEngineState *state) {
    // std::cout << "MultiPV: " << state->thinkerInterface()->get_multi_pv() << " variations" << std::endl;
    // std::cout << "Threads: " << state->thinkerInterface()->get_num_threads() << " threads" << std::endl;
    // std::cout << "Hash: " << state->thinkerInterface()->get_cache_size_kb() << " kilobytes" << std::endl;
  }
};

class QuitTask : public Task {
 public:
  void start(UciEngineState *state) {
    exit(0);
  }
};

class StopTask : public Task {
 public:
  void start(UciEngineState *state) {
    state->stopThinking.store(true);
  }
};

class NewGameTask : public Task {
 public:
  void start(UciEngineState *state) {
    state->tt_->new_search();
  }
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_TRIVIALTASKS_H
