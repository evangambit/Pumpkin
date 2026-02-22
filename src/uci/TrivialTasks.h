#ifndef PUMPKIN_UCI_TRIVIALTASKS_H
#define PUMPKIN_UCI_TRIVIALTASKS_H

// model.o
extern const char model_bin[];
extern unsigned int model_bin_len;

extern const char qst_bin[];
extern unsigned int qst_bin_len;

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <fstream>

#include "Task.h"
#include "../eval/Evaluator.h"
#include "../eval/nnue/NnueEvaluator.h"
#include "../eval/pst/PieceSquareEvaluator.h"
#include "../eval/qst/QstEvaluator.h"
#include "../game/movegen/movegen.h"

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

class ProbeTask : public Task {
 public:
  ProbeTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() < 2) {
      std::cout << "Error: probe command requires a move argument." << std::endl;
      return;
    }
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
        std::cout << "  Value: " << "cp " << entry.value;
      }
      std::cout << "  Depth: " << entry.depth;
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
      Threats threats;
      create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
      if (pos.turn_ == Color::WHITE) {
        ColoredEvaluation<WHITE> eval = evaluate<WHITE>(pos.evaluator_, pos, threats);
        std::cout << eval.value << " (white) (" << pos.evaluator_->to_string() << ")" << std::endl;
      } else {
        ColoredEvaluation<BLACK> eval = evaluate<BLACK>(pos.evaluator_, pos, threats);
        std::cout << eval.value << " (black) (" << pos.evaluator_->to_string() << ")" << std::endl;
      }
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
      Threats threats;
      create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
      if (pos.turn_ == Color::BLACK) {
        move->score = -evaluate<BLACK>(pos.evaluator_, pos, threats).value;
      } else {
        move->score = evaluate<WHITE>(pos.evaluator_, pos, threats).value;
      }
      ez_undo(&pos);
    }
    if (pos.turn_ == Color::BLACK) {
      std::sort(moves, end, [](ExtMove a, ExtMove b) { return a.score < b.score; });
    } else {
      std::sort(moves, end, [](ExtMove a, ExtMove b) { return a.score > b.score; });
    }
    for (ExtMove* move = moves; move != end; ++move) {
      std::cout << move->uci() << ": " << move->score << " (white) (" << pos.evaluator_->to_string() << ")" << std::endl;
    }
  }
 private:
  std::deque<std::string> command;
};

struct FenErrorResult {
  std::string fen;
  float expected;
  float actual;
  float error;
};

struct FenErrorTask : public Task {
  FenErrorTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() < 2) {
      std::cout << "Error: fenerror command requires a file argument." << std::endl;
      return;
    }
    std::string filename = command.at(1);
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cout << "Error: could not open file " << filename << std::endl;
      return;
    }
    
    std::vector<FenErrorResult> results;
    std::string line;
    while (std::getline(file, line)) {
      size_t pos = line.find('|');
      if (pos == std::string::npos) continue;
      std::string fen = line.substr(0, pos);
      float expected = std::stof(line.substr(pos + 1));
      
      Position p(fen);
      p.set_listener(state->position.evaluator_->clone());
      Threats threats;
      create_threats(p.pieceBitboards_, p.colorBitboards_, &threats);
      ColoredEvaluation<WHITE> eval = evaluate<WHITE>(p.evaluator_, p, threats);
      
      float actual = NNUE::sigmoid(eval.value / float(1 << NNUE::SCALE_SHIFT));
      float error = std::abs(expected - actual);
      results.push_back({fen, expected, actual, error});
    }
    
    std::sort(results.begin(), results.end(), [](const FenErrorResult& a, const FenErrorResult& b) {
      return a.error > b.error;
    });
    
    int limit = std::min(10, (int)results.size());
    for (int i = 0; i < limit; ++i) {
      std::cout << results[i].fen << " Expected: " << results[i].expected 
                << " Actual: " << results[i].actual << " Error: " << results[i].error << std::endl;
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
    command.pop_front();
    std::string evaluatorName = command.at(0);
    command.pop_front();
    if (evaluatorName == "simple") {
      state->position.set_listener(std::make_shared<SimpleEvaluator>());
      std::cout << "Evaluator set to simple." << std::endl;
    } else if (evaluatorName == "pst") {
      state->position.set_listener(std::make_shared<PieceSquareEvaluator>());
      std::cout << "Evaluator set to pst." << std::endl;
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
        std::cout << "Model loaded successfully." << std::endl;
      } else {
        std::istringstream f(std::string(model_bin, model_bin_len));
        nnue_model->load(f);
      }
      state->position.set_listener(std::make_shared<NNUE::NnueEvaluator<int16_t>>(nnue_model));
      std::cout << "Evaluator set to nnue." << std::endl;
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
        std::cout << "Model loaded successfully." << std::endl;
      } else {
        std::istringstream f(std::string(model_bin, model_bin_len));
        nnue_model->load(f);
      }
      state->position.set_listener(std::make_shared<NNUE::NnueEvaluator<float>>(nnue_model));
      std::cout << "Evaluator set to nnue." << std::endl;
    } else if (evaluatorName == "qst") {
      auto qst = std::make_shared<QstEvaluator>();
      if (command.size() > 0) {
        std::string modelFile = command.at(0);
        command.pop_front();
        qst->load(modelFile);
      } else {
        std::istringstream f(std::string(qst_bin, qst_bin_len));
        qst->load(f);
      }
      state->position.set_listener(qst);
      std::cout << "Evaluator set to qst." << std::endl;
    } else {
      std::cout << "Error: unrecognized evaluator name \"" << evaluatorName << "\"" << std::endl;
    }
    state->tt_->clear();
  }
 private:
  std::deque<std::string> command;
};

class NnueEvalDebugTask : public Task {
 public:
  NnueEvalDebugTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    std::shared_ptr<NNUE::Nnue<float>> nnue_model = std::make_shared<NNUE::Nnue<float>>();
    if (command.size() == 0) {
      std::cout << "Error: nnue eval debug command requires a model file." << std::endl;
      return;
    }
    command.pop_front();
    std::string modelFile = command.at(0);
    command.pop_front();
    std::ifstream f(modelFile, std::ios::binary);
    if (!f) {
      std::cout << "Error: could not open model file \"" << modelFile << "\"" << std::endl;
      return;
    }
    nnue_model->load(f);
    Position pos = state->position;
    auto evaluator = std::make_shared<NNUE::NnueEvaluator<float>>(nnue_model);
    pos.set_listener(evaluator);
    Threats threats;
    create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);

    float value;
    if (pos.turn_ == Color::WHITE) {
      ColoredEvaluation<WHITE> eval = evaluate<WHITE>(evaluator, pos, threats);
      value = float(eval.value) / float(1 << NNUE::SCALE_SHIFT);
    } else {
      ColoredEvaluation<BLACK> eval = evaluate<BLACK>(evaluator, pos, threats);
      value = float(eval.value) / float(1 << NNUE::SCALE_SHIFT);
    }
    std::streamsize ss = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Emb: " << nnue_model->embWeights[0][0] << ", " << nnue_model->embWeights[0][1] << std::endl;
    std::cout << "White Acc: " << nnue_model->whiteAcc.data[0] << ", " << nnue_model->whiteAcc.data[1] << std::endl;
    std::cout << "Black Acc: " << nnue_model->blackAcc.data[0] << ", " << nnue_model->blackAcc.data[1] << std::endl;
    std::cout << "layer1: " << nnue_model->layer1.data[0] << ", " << nnue_model->layer1.data[1] << std::endl;
    std::cout << "bias1: " << nnue_model->bias1.data[0] << " " << nnue_model->bias1.data[1] << std::endl;
    std::cout << "hidden1: " << nnue_model->hidden1.data[0] << ", " << nnue_model->hidden1.data[1] << ", " << nnue_model->hidden1.data[2] << ", " << nnue_model->hidden1.data[3] << ", " << nnue_model->hidden1.data[4] << ", " << nnue_model->hidden1.data[5] << ", " << nnue_model->hidden1.data[6] << ", " << nnue_model->hidden1.data[7] << std::endl;
    std::cout << "layer2: " << nnue_model->layer2.data[0] << ", " << nnue_model->layer2.data[1] << std::endl;
    std::cout << "bias2: " << nnue_model->bias2.data[0] << std::endl;
    std::cout << value << " (" << (pos.turn_ == Color::WHITE ? "white" : "black") << ")" << std::endl;
    std::cout.precision(ss);
  }
 private:
  std::deque<std::string> command;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_TRIVIALTASKS_H
