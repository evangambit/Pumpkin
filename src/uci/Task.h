#ifndef PUMPKIN_UCI_TASK_H
#define PUMPKIN_UCI_TASK_H

#include "../eval/nnue/NnueEvaluator.h"
#include "../eval/pst/PieceSquareEvaluator.h"
#include "../eval/byhand/byhand.h"
#include "../search/transposition_table.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

extern const char model_bin[];
extern unsigned int model_bin_len;

extern const char byhand_bin[];
extern unsigned int byhand_bin_len;

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

  virtual ~Task() = default;

  virtual bool is_running() {
    return false;
  }

  bool is_slow() {
    return false;
  }
};


struct UciEngineState {
  UciEngineState()
    : tt_(std::make_shared<TranspositionTable>(/* megabytes= */100)),
      moveOverheadMs(50),
      numThreads(1),
      multiPV(1) {
    this->position = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    // std::shared_ptr<NNUE::Nnue> nnue_model = std::make_shared<NNUE::Nnue>();
    // std::istringstream f(std::string(model_bin, model_bin_len));
    // nnue_model->load(f);
    #ifndef DEFAULT_EVALUATOR
    this->position.set_listener(std::make_shared<PieceSquareEvaluator>());
    #elif DEFAULT_EVALUATOR == 'b'
    auto evaluator = std::make_shared<ByHand::ByHandEvaluator>();
    std::istringstream f(std::string(byhand_bin, byhand_bin_len));
    evaluator->load_from_stream(f);
    this->position.set_listener(evaluator);
    #elif DEFAULT_EVALUATOR == 'n'
    auto nnue_model = std::make_shared<NNUE::Nnue>();
    std::istringstream f(std::string(model_bin, model_bin_len));
    nnue_model->load(f);
    this->position.set_listener(std::make_shared<NNUE::NnueEvaluator>(nnue_model));
    #else
    static_assert(false, "Invalid DEFAULT_EVALUATOR");
    #endif
  }

  std::mutex mutex;
  std::condition_variable condVar;
  // Signals input and worker loops to stop waiting and exit on shutdown/EOF.
  std::atomic<bool> shuttingDown = false;

  std::deque<std::shared_ptr<Task>> taskQueue;
  SpinLock taskQueueLock;
  std::shared_ptr<Task> currentTask;
  std::shared_ptr<std::atomic<bool>> stopThinking = std::make_shared<std::atomic<bool>>(false);

  std::shared_ptr<TranspositionTable> tt_;

  unsigned moveOverheadMs;
  unsigned numThreads;
  unsigned multiPV;
  Position position;
  std::string name;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_TASK_H
