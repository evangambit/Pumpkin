#ifndef PUMPKIN_UCI_TASK_H
#define PUMPKIN_UCI_TASK_H

#include "../eval/PieceSquareEvaluator.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace ChessEngine {

void invalid(const std::string& command) {
  std::cout << "Invalid use of " << repr(command) << " command" << std::endl;
}

void invalid(const std::string& command, const std::string& message) {
  std::cout << "Invalid use of " << repr(command) << " command (" << message << ")" << std::endl;
}

struct SpinLock {
  std::atomic<bool> lock_ = {false};
  void lock() { while(lock_.exchange(true)); }
  void unlock() { lock_.store(false); }
};

struct UciEngineState;

class Task {
 public:
  virtual void start(UciEngineState *state) = 0;

  virtual bool is_running() {
    return false;
  }

  bool is_slow() {
    return false;
  }
};


struct UciEngineState {
  UciEngineState() : tt_(std::make_shared<TranspositionTable>(100'000)),
                      moveOverheadMs(50),
                      numThreads(1),
                      multiPV(1),
                      evaluator(std::make_shared<PieceSquareEvaluator>()) {}

  std::mutex mutex;
  std::condition_variable condVar;

  std::deque<std::shared_ptr<Task>> taskQueue;
  SpinLock taskQueueLock;
  std::shared_ptr<Task> currentTask;
  std::atomic<bool> stopThinking{false};

  std::shared_ptr<TranspositionTable> tt_;

  unsigned moveOverheadMs;
  unsigned numThreads;
  unsigned multiPV;
  Position position;
  std::shared_ptr<EvaluatorInterface> evaluator;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_TASK_H
