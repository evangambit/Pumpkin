#ifndef PUMPKIN_UCI_TRIVIALTASKS_H
#define PUMPKIN_UCI_TRIVIALTASKS_H

// model.o
extern const char model_bin[];
extern unsigned int model_bin_len;

extern const char qst_bin[];
extern unsigned int qst_bin_len;

extern const char byhand_bin[];
extern unsigned int byhand_bin_len;

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>
#include <fstream>

#include "Task.h"
#include "../eval/Evaluator.h"
#include "../eval/byhand/byhand.h"
#include "../eval/nnue/NnueEvaluator.h"
#include "../eval/pst/PieceSquareEvaluator.h"
#include "../game/movegen/movegen.h"

namespace ChessEngine {

template <Color COLOR>
NegamaxResult<COLOR> ez_qsearch(const Position& pos) {
  // qsearch(thread, alpha, beta, plyFromRoot, 0, frame, stopThinking)
  const unsigned int multiPV = 1;
  const unsigned int numThreads = 1;
  auto tt = std::make_shared<TranspositionTable>(/* megabytes= */1);
  GoCommand command;
  const bool isTimeSensitive = false;
  auto thread = std::make_shared<SearchThread>(
    /*id=*/ 0,
    /*position=*/ pos,
    /*shared=*/ std::make_shared<SharedSearchThreadState>(command, multiPV, numThreads, isTimeSensitive, std::chrono::high_resolution_clock::time_point::max(), tt.get())
  );
  thread->root_frame()->inCheck = can_enemy_attack<COLOR>(
    thread->position_,
    lsb_i_promise_board_is_not_empty(thread->position_.pieceBitboards_[coloredPiece<COLOR, Piece::KING>()])
  );
  std::atomic<bool> stopThinking = false;
  return qsearch<COLOR>(
      thread.get(),
      ColoredEvaluation<COLOR>(kMinEval),
      ColoredEvaluation<COLOR>(kMaxEval),
      0, 0, thread->root_frame(), &stopThinking);
}

ColoredEvaluation<WHITE> foo(const Position& pos, bool q) {
  Threats threats;
  create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
  if (pos.turn_ == Color::WHITE) {
    if (q) {
      auto r = ez_qsearch<WHITE>(pos);
      std::cout << r.evaluation << " (w) " << r.bestMove.uci() << std::endl;
      return r.evaluation;
    } else {
      ColoredEvaluation<WHITE> e = evaluate<WHITE>(pos.evaluator_, pos, threats, 0, ColoredEvaluation<WHITE>(kMinEval), ColoredEvaluation<WHITE>(kMaxEval));
      std::cout << e << " (w)" << std::endl;
      return e;
    }
  } else {
    if (q) {
      auto r = ez_qsearch<BLACK>(pos);
      std::cout << r.evaluation << " (b) " << r.bestMove.uci() << std::endl;
      return -r.evaluation;
    } else {
      ColoredEvaluation<BLACK> e = evaluate<BLACK>(pos.evaluator_, pos, threats, 0, ColoredEvaluation<BLACK>(kMinEval), ColoredEvaluation<BLACK>(kMaxEval));
      std::cout << e << " (b)" << std::endl;
      return -e;
    }
  }
}

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

class ProbeTask : public Task {
 public:
  ProbeTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    command.pop_front();
    Position pos = state->position;
    while (command.size() > 0) {
      std::string moveStr = command.at(0);
      command.pop_front();
      Move move = uci_to_move(pos, moveStr);
      if (move == kNullMove) {
        std::cout << "Error: invalid move \"" << moveStr << "\"" << std::endl;
        return;
      }
      ez_make_move(&pos, move);
    }
    TTEntry entry;
    size_t counter = 0;
    while (state->tt_->probe(pos.currentState_.hash, entry) && (counter++ < 10)) {
      if (pos.turn_ == Color::BLACK) {
        // Print bounds/values from white's perspective.
        entry = entry.flip();
      }
      std::cout << entry.bestMove.uci();
      if (entry.value <= kLongestForcedMate) {
        std::cout << "  Value: " << "mate " << -(entry.value - kCheckmate + 1) / 2;
      } else if (entry.value >= -kLongestForcedMate) {
        std::cout << "  Value: " << "mate " << -(entry.value + kCheckmate - 1) / 2;
      } else {
        std::cout << "  Value: " << "wcp " << entry.value;
      }
      std::cout << "  Depth: " << int(entry.depth);
      std::cout << "  Bound: " << bound_type_to_string(entry.bound);
      std::cout << "  Hash: " << pos.currentState_.hash;
      std::cout << std::endl;
      if (entry.bestMove == kNullMove) {
        break;
      }
      ez_make_move(&pos, entry.bestMove);
    }
  }
 private:
  std::deque<std::string> command;
};

struct EvalTask : public Task {
  EvalTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    Position pos = state->position;
    command.pop_front();
    bool q = false;
    if (command.size() > 0 && command.at(0) == "q") {
      q = true;
      command.pop_front();
    }
    bool printAllChildren = false;
    while (command.size() > 0) {
      std::string moveStr = command.at(0);
      command.pop_front();
      if (moveStr == "*") {
        printAllChildren = true;
        break;
      }
      Move move = uci_to_move(pos, moveStr);
      if (move == kNullMove) {
        std::cout << "Error: invalid move \"" << moveStr << "\"" << std::endl;
        return;
      }
      ez_make_move(&pos, move);
    }
    if (!printAllChildren) {
      foo(pos, q);
      return;
    }
    ExtMove moves[kMaxNumMoves];
    ExtMove* end;
    if (pos.turn_ == Color::BLACK) {
      end = compute_legal_moves<Color::BLACK>(&pos, &(moves[0]));
    } else {
      end = compute_legal_moves<Color::WHITE>(&pos, &(moves[0]));
    }
    for (ExtMove* move = moves; move != end; ++move) {
      ez_make_move(&pos, move->move);
      if (pos.turn_ == Color::BLACK) {
        move->score = -foo(pos, q).value;
      } else {
        move->score = foo(pos, q).value;
      }
      ez_undo(&pos);
    }
    if (pos.turn_ == Color::BLACK) {
      std::sort(moves, end, [](ExtMove a, ExtMove b) { return a.score < b.score; });
    } else {
      std::sort(moves, end, [](ExtMove a, ExtMove b) { return a.score > b.score; });
    }
    for (ExtMove* move = moves; move != end; ++move) {
      ez_make_move(&pos, move->move);
      foo(pos, q);
      ez_undo(&pos);
    }
  }
 private:
  std::deque<std::string> command;
};

struct MoveTask : public Task {
  MoveTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    command.pop_front();
    while (command.size() > 0) {
      std::string moveStr = command.front();
      command.pop_front();
      Move move = uci_to_move(state->position, moveStr);
      if (move == kNullMove) {
        std::cout << "Error: invalid move \"" << moveStr << "\"" << std::endl;
        return;
      }
      ez_make_move(&state->position, move);
    }
  }

 private:
  std::deque<std::string> command;
};

class PrintFenTask : public Task {
 public:
  void start(UciEngineState *state) {
    std::cout << state->position.fen() << "|" << state->position.currentState_.hash << std::endl;
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
    std::cout << "Threads: " << state->numThreads << " threads" << std::endl;
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
    if (state->stopThinking) {
      state->stopThinking->store(true);
    }
  }
};

class NewGameTask : public Task {
 public:
  void start(UciEngineState *state) {
    // state->tt_->new_search();  // Faster, but less guaranteed to let us reproduce games.
    state->tt_->clear();
  }
};

class SetEvaluatorTask : public Task {
 public:
  SetEvaluatorTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() < 2) {
      std::cout << state->position.evaluator_->to_string() << std::endl;
      return;
    }
    command.pop_front();
    std::string evaluatorName = command.at(0);
    command.pop_front();
    if (evaluatorName == "simple") {
      state->position.set_listener(std::make_shared<SimpleEvaluator>());
      std::cout << "info string evaluator set to simple." << std::endl;
    } else if (evaluatorName == "pst") {
      state->position.set_listener(std::make_shared<PieceSquareEvaluator>());
      std::cout << "info string evaluator set to pst." << std::endl;
    } else if (evaluatorName == "byhand") {
      auto evaluator = std::make_shared<ByHand::ByHandEvaluator>();
      state->position.set_listener(evaluator);
      std::cout << "info string evaluator set to byhand." << std::endl;
      if (command.size() > 0) {
        std::string modelFile = command.at(0);
        command.pop_front();
        std::ifstream f(modelFile, std::ios::binary);
        if (!f) {
          std::cout << "Error: could not open model file \"" << modelFile << "\"" << std::endl;
          return;
        }
        evaluator->load_from_stream(f);
      } else {
        std::istringstream f(std::string(byhand_bin, byhand_bin_len));
        evaluator->load_from_stream(f);
      }
    } else if (evaluatorName == "nnue") {
      std::shared_ptr<NNUE::Nnue<int16_t>> nnue_model = std::make_shared<NNUE::Nnue<int16_t>>();
      if (command.size() > 0) {
        std::string modelFile = command.at(0);
        command.pop_front();
        std::ifstream f(modelFile, std::ios::binary);
        if (!f) {
          std::cout << "Error: could not open model file \"" << modelFile << "\"" << std::endl;
          return;
        }
        nnue_model->load(f);
        std::cout << "info string Model loaded successfully." << std::endl;
      } else {
        std::istringstream f(std::string(model_bin, model_bin_len));
        nnue_model->load(f);
      }
      state->position.set_listener(std::make_shared<NNUE::NnueEvaluator<int16_t>>(nnue_model));
      std::cout << "info string Evaluator set to nnue." << std::endl;
    } else if (evaluatorName == "nnuef") {
      std::shared_ptr<NNUE::Nnue<float>> nnue_model = std::make_shared<NNUE::Nnue<float>>();
      if (command.size() > 0) {
        std::string modelFile = command.at(0);
        command.pop_front();
        std::ifstream f(modelFile, std::ios::binary);
        if (!f) {
          std::cout << "Error: could not open model file \"" << modelFile << "\"" << std::endl;
          return;
        }
        nnue_model->load(f);
        std::cout << "info string Model loaded successfully." << std::endl;
      } else {
        std::istringstream f(std::string(model_bin, model_bin_len));
        nnue_model->load(f);
      }
      state->position.set_listener(std::make_shared<NNUE::NnueEvaluator<float>>(nnue_model));
      std::cout << "info string Evaluator set to nnue." << std::endl;
    } else {
      std::cout << "Error: unrecognized evaluator name \"" << evaluatorName << "\"" << std::endl;
      exit(1);
    }
    state->tt_->clear();
  }
 private:
  std::deque<std::string> command;
};

class IncrementSearchHyperParamTask : public Task {
 public:
  IncrementSearchHyperParamTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    command.pop_front();
    if (command.size() != 2) {
      std::cout << "Error: increment_search command requires exactly 2 arguments: <param_name> <delta>" << std::endl;
      exit(1);
    }
    std::string paramName = command.at(0);
    int delta = std::stoi(command.at(1));
    auto& p = state->searchHyperParams;

    if (paramName == "lmr_pv_a") {
      apply_fixed_point(p.lmr_pv_a, delta);
    } else if (paramName == "lmr_pv_b") {
      apply_fixed_point(p.lmr_pv_b, delta);
    } else if (paramName == "lmr_null_a") {
      apply_fixed_point(p.lmr_null_a, delta);
    } else if (paramName == "lmr_null_b") {
      apply_fixed_point(p.lmr_null_b, delta);
    } else if (paramName == "singular_margin") {
      apply_int(p.singular_margin, delta);
    } else if (paramName == "razoring_margin") {
      apply_int(p.razoring_margin, delta);
    } else if (paramName == "futility_margin") {
      apply_int(p.futility_margin, delta);
    } else if (paramName == "null_move_pruning_depth_reduction") {
      apply_int(p.null_move_pruning_depth_reduction, delta);
    } else {
      std::cout << "Error: unknown search hyper param '" << paramName << "'" << std::endl;
      exit(1);
    }
  }
 private:
  static void apply_int(int& value, int delta) {
    value += delta;
  }
  static void apply_fixed_point(FixedPoint<int32_t, 8>& value, int delta) {
    value = value + FixedPoint<int32_t, 8>(delta);
  }
  static void apply_fixed_point(FixedPoint<int32_t, 8>& value, char op, int delta) {
    if (op == '+') {
      value = value + FixedPoint<int32_t, 8>(delta);
    } else {
      value = value * FixedPoint<int32_t, 8>(delta) / FixedPoint<int32_t, 8>(100);
    }
  }
  std::deque<std::string> command;
};

class DumpWeightsTask : public Task {
 public:
  DumpWeightsTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (state->position.evaluator_->to_string() != "ByHandEvaluator") {
      std::cout << "Error: dumpweights only works with byhand evaluator." << std::endl;
      return;
    }
    command.pop_front();
    if (command.size() < 1) {
      std::cout << "Error: dumpweights requires a filename argument." << std::endl;
      return;
    }
    std::string filename = command.at(0);
    auto evaluator = std::dynamic_pointer_cast<ByHand::ByHandEvaluator>(state->position.evaluator_);

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      std::cout << "Error: could not open file \"" << filename << "\" for writing." << std::endl;
      return;
    }

    const float scale = static_cast<float>(1 << NNUE::SCALE_SHIFT);

    // Write weights matrix (2 x EF_COUNT)
    write_matrix(out, "weights", 2, ByHand::EF::EF_COUNT, [&](size_t i, size_t j) {
      return static_cast<float>(evaluator->weights(i, j)) / scale;
    });

    // Write bias vector (2)
    write_vector(out, "bias", 2, [&](size_t i) {
      return static_cast<float>(evaluator->bias[i]) / scale;
    });

    // Write pst_late matrix (6 x 64)
    write_matrix(out, "pst_late", 6, 64, [&](size_t piece, size_t sq) {
      return static_cast<float>(evaluator->pstWeights[piece * 64 + sq + 8 * 64]) / scale;
    });

    // Write pst_early matrix (6 x 64)
    write_matrix(out, "pst_early", 6, 64, [&](size_t piece, size_t sq) {
      return static_cast<float>(evaluator->pstWeights[piece * 64 + sq + 64]) / scale;
    });

    out.close();
    std::cout << "Weights dumped to " << filename << std::endl;
  }
 private:
  std::deque<std::string> command;

  static void write_padded_name(std::ofstream& out, const std::string& name) {
    char buf[16];
    std::memset(buf, ' ', 16);
    std::memcpy(buf, name.c_str(), std::min(name.size(), size_t(16)));
    out.write(buf, 16);
  }

  template<typename Func>
  static void write_matrix(std::ofstream& out, const std::string& name,
                           uint32_t rows, uint32_t cols, Func getValue) {
    write_padded_name(out, name);
    uint32_t degree = 2;
    out.write(reinterpret_cast<const char*>(&degree), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
    for (uint32_t i = 0; i < rows; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
        float val = getValue(i, j);
        out.write(reinterpret_cast<const char*>(&val), sizeof(float));
      }
    }
  }

  template<typename Func>
  static void write_vector(std::ofstream& out, const std::string& name,
                           uint32_t size, Func getValue) {
    write_padded_name(out, name);
    uint32_t degree = 1;
    out.write(reinterpret_cast<const char*>(&degree), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; ++i) {
      float val = getValue(i);
      out.write(reinterpret_cast<const char*>(&val), sizeof(float));
    }
  }
};

struct AnalyzeFens : public Task {
  AnalyzeFens(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    // analyzefens <filename> <depth> <limit>
    // Dumps to out.txt
    // Writes <fen>|<eval@depth=1>|<eval@depth=2>|... for each fen in the file, up to the specified depth.
    if (command.front() != "analyzefens") {
      std::cout << "Error: expected analyzefens command." << std::endl;
      return;
    }
    command.pop_front();
    if (command.size() != 3) {
      std::cout << "Error: analyzefens command requires a filename, depth, and limit argument." << std::endl;
      return;
    }
    std::string filename = command.at(0);
    int depth = std::stoi(command.at(1));
    int limit = std::stoi(command.at(2));

    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cout << "Error: could not open file " << filename << std::endl;
      return;
    }

    if (depth < 1) {
      std::cout << "Error: analyzefens depth must be at least 1." << std::endl;
      return;
    }
    if (limit < 0) {
      std::cout << "Error: analyzefens limit must be non-negative." << std::endl;
      return;
    }

    std::ofstream out("out.txt");
    if (!out.is_open()) {
      std::cout << "Error: could not open out.txt for writing." << std::endl;
      return;
    }

    std::vector<std::string> fens;
    fens.reserve(limit);
    std::string line;
    while (fens.size() < static_cast<size_t>(limit) && std::getline(file, line)) {
      size_t separator = line.find('|');
      std::string fen = separator == std::string::npos ? line : line.substr(0, separator);
      remove_excess_whitespace(&fen);
      if (fen.empty()) {
        continue;
      }

      fens.push_back(fen);
    }

    if (fens.empty()) {
      std::cout << "Analyzed 0 FENs to depth " << depth << " into out.txt" << std::endl;
      return;
    }

    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const size_t numWorkers = std::min<size_t>(
      fens.size(),
      hardwareThreads > 0 ? hardwareThreads : 4
    );
    const size_t ttSizeMb = std::max<size_t>(1, state->tt_->kb_size() / 1024);
    const size_t ttSizeMbPerWorker = std::max<size_t>(1, ttSizeMb / numWorkers);
    const size_t progressInterval = std::max<size_t>(1, fens.size() / 20);

    std::atomic<size_t> nextFenIndex{0};
    std::atomic<size_t> analyzed{0};
    std::mutex ioMutex;
    std::vector<std::thread> workers;
    workers.reserve(numWorkers);

    for (size_t workerId = 0; workerId < numWorkers; ++workerId) {
      workers.emplace_back([&, workerId]() {
        auto tt = std::make_shared<TranspositionTable>(ttSizeMbPerWorker);

        while (true) {
          size_t fenIndex = nextFenIndex.fetch_add(1);
          if (fenIndex >= fens.size()) {
            break;
          }

          const std::string& fen = fens[fenIndex];

          Position pos(fen);
          pos.set_listener(state->position.evaluator_->clone());

          GoCommand goCommand;
          goCommand.pos = pos;
          goCommand.depthLimit = depth;

          auto shared = std::make_shared<SharedSearchThreadState>(
            goCommand,
            /* multiPV=*/1,
            /* numThreads=*/1,
            /* isTimeSensitive=*/false,
            std::chrono::high_resolution_clock::time_point::max(),
            tt.get()
          );
          shared->search_hyper_params = state->searchHyperParams;

          SearchThread searchThread(
            /* thread id=*/static_cast<int>(workerId),
            pos,
            shared
          );

          std::vector<Evaluation> evals;
          evals.reserve(depth);
          std::atomic<bool> neverStop{false};
          SearchResult<Color::WHITE> finalResult = colorless_search(
            &searchThread,
            &neverStop,
            [&evals](int completedDepth, SearchResult<Color::WHITE> result, uint64_t, uint64_t) {
              if (evals.size() < static_cast<size_t>(completedDepth)) {
                evals.resize(completedDepth);
              }
              evals[completedDepth - 1] = result.evaluation.value;
            }
          );

          if (evals.empty()) {
            evals.push_back(finalResult.evaluation.value);
          }

          std::ostringstream row;
          row << fen;
          for (Evaluation eval : evals) {
            row << "|" << eval;
          }
          row << std::endl;

          const size_t completed = analyzed.fetch_add(1) + 1;
          const bool shouldPrintProgress =
            completed == fens.size() ||
            (progressInterval > 0 && completed % progressInterval == 0);

          std::lock_guard<std::mutex> lock(ioMutex);
          out << row.str();
          if (shouldPrintProgress && completed < fens.size()) {
            std::cout << "Analyzed " << completed << " / " << fens.size() << " FENs..." << std::endl;
          }
        }
      });
    }

    for (std::thread& worker : workers) {
      worker.join();
    }

    std::cout << "Analyzed " << analyzed.load() << " FENs to depth " << depth << " into out.txt" << std::endl;

  }
 private:
  std::deque<std::string> command;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_TRIVIALTASKS_H
