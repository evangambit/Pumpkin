#ifndef PUMPKIN_UCI_SETOPTIONTASK_H
#define PUMPKIN_UCI_SETOPTIONTASK_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "Task.h"

namespace ChessEngine {

bool does_pattern_match(const std::deque<std::string>& text, const std::vector<std::string>& pattern) {
  if (text.size() != pattern.size()) {
    return false;
  }
  for (size_t i = 0; i < text.size(); ++i) {
    if (pattern[i] != "*" && text[i] != pattern[i]) {
      return false;
    }
  }
  return true;
}

class SetOptionTask : public Task {
 public:
  SetOptionTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    assert(command.size() > 0 && command[0] == "setoption");
    command.pop_front();
    if (command.size() == 0 || command[0] != "name") {
      invalid(join(command, " "));
      return;
    }
    command.pop_front();

    if(does_pattern_match(command, {"Move", "Overhead", "value", "*"})) {
      state->moveOverheadMs = std::stoi(command[3]);
    } else if (does_pattern_match(command, {"Clear", "Hash"})) {
      state->tt_->clear();
      return;
    } else if (does_pattern_match(command, {"MultiPV", "value", "*"})) {
      int multiPV;
      try {
        multiPV = std::stoi(command[2]);
        if (multiPV <= 0) {
          throw std::invalid_argument("Value must be at least 1");
        }
      } catch (std::invalid_argument&) {
        std::cout << "Value must be an integer" << std::endl;
        return;
      }
      if (multiPV < 1) {
        std::cout << "Value must be positive" << std::endl;
        return;
      }
      state->multiPV = multiPV;
      return;
    } else if (does_pattern_match(command, {"Threads", "value", "*"})) {
      int numThreads;
      try {
        numThreads = std::stoi(command[2]);
        if (numThreads <= 0) {
          throw std::invalid_argument("Value must be at least 1");
        }
      } catch (std::invalid_argument&) {
        std::cout << "Value must be an integer" << std::endl;
        return;
      }
      if (numThreads < 1) {
        std::cout << "Value must be positive" << std::endl;
        return;
      }
      state->numThreads = numThreads;
      return;
    } else if (does_pattern_match(command, {"Hash", "value", "*"})) {
      int cacheSizeMb;
      try {
        cacheSizeMb = std::stoi(command[2]);
        if (cacheSizeMb <= 0) {
          throw std::invalid_argument("Value must be at least 1");
        }
      } catch (std::invalid_argument&) {
        std::cout << "Value must be an integer" << std::endl;
        return;
      }
      state->tt_->resize(cacheSizeMb);
      return;
    } else if (does_pattern_match(command, {"SyzygyPath", "value", "*"})) {
      // TODO
      return;
    } else if (does_pattern_match(command, {"UCI_ShowWDL", "value", "*"})) {
      // TODO
      return;
    } else if (does_pattern_match(command, {"Ponder", "value", "*"})) {
      // TODO
      return;
    } else {
      std::cout << "Unrecognized option \"" << join(command, " ") << "\"" << std::endl;
    }
  }
 private:

  std::deque<std::string> command;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_SETOPTIONTASK_H
