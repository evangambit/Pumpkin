#ifndef PUMPKIN_UCI_TRIVIALTASKS_H
#define PUMPKIN_UCI_TRIVIALTASKS_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <fstream>

#include "Task.h"
#include "../eval/PieceSquareEvaluator.h"
#include "../eval/evaluator.h"
#include "../eval/nnue/NnueEvaluator.h"

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
    std::cout << "MultiPV: " << state->multiPV << " variations" << std::endl;
    // std::cout << "Threads: " << state->thinkerInterface()->get_num_threads() << " threads" << std::endl;
    std::cout << "Hash: " << state->tt_->kb_size() << " kilobytes" << std::endl;
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

class SetEvaluatorTask : public Task {
 public:
  SetEvaluatorTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() < 2) {
      std::cout << "Error: evaluator command requires an argument." << std::endl;
      return;
    }
    std::string evaluatorName = command.at(1);
    if (evaluatorName == "simple") {
      state->evaluator = std::make_shared<SimpleEvaluator>();
      std::cout << "Evaluator set to simple." << std::endl;
    } else if (evaluatorName == "pst") {
      state->evaluator = std::make_shared<PieceSquareEvaluator>();
      std::cout << "Evaluator set to pst." << std::endl;
    } else if (evaluatorName == "nnue") {
      std::shared_ptr<NNUE::Nnue> nnue_model = std::make_shared<NNUE::Nnue>();
      std::ifstream f("model.bin", std::ios::binary);
      nnue_model->load(f);
      state->evaluator = std::make_shared<NNUE::NnueEvaluator>(nnue_model);
      std::cout << "Evaluator set to nnue." << std::endl;
    } else {
      std::cout << "Error: unrecognized evaluator name \"" << evaluatorName << "\"" << std::endl;
    }
  }
 private:
  std::deque<std::string> command;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_TRIVIALTASKS_H
