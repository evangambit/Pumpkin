#ifndef PUMPKIN_UCI_POSITIONTASK_H
#define PUMPKIN_UCI_POSITIONTASK_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "Task.h"

namespace ChessEngine {

class PositionTask : public Task {
 public:
  PositionTask(std::deque<std::string> command) : command(command) {}
  void start(UciEngineState *state) {
    if (command.size() < 2) {
      invalid(join(command, " "));
      return;
    }
    std::shared_ptr<EvaluatorInterface> evaluator = state->position.evaluator_;
    size_t i;
    if (command[1] == "startpos") {
      state->position = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
      i = 2;
    } else if (command.size() >= 8 && command[1] == "fen") {
      std::vector<std::string> fen(command.begin() + 2, command.begin() + 8);
      i = 8;
      state->position = Position(join(fen, " "));
    } else {
      invalid(join(command, " "));
      return;
    }
    state->position.set_listener(evaluator->clone());
    if (i == command.size()) {
      return;
    }
    if (command[i] != "moves") {
      invalid(join(command, " "));
      return;
    }
    while (++i < command.size()) {
      std::string uciMove = command[i];
      ExtMove moves[kMaxNumMoves];
      ExtMove *end;
      if (state->position.turn_ == Color::WHITE) {
        end = compute_legal_moves<Color::WHITE>(&state->position, moves);
      } else {
        end = compute_legal_moves<Color::BLACK>(&state->position, moves);
      }
      bool foundMove = false;
      for (ExtMove *move = moves; move < end; ++move) {
        if (move->move.uci() == uciMove) {
          foundMove = true;
          if (state->position.turn_ == Color::WHITE) {
            make_move<Color::WHITE>(&state->position, move->move);
          } else {
            make_move<Color::BLACK>(&state->position, move->move);
          }
          break;
        }
      }
      if (!foundMove) {
        std::cout << "Could not find move " << repr(uciMove) << std::endl;
        return;
      }
    }
  }
 private:
  std::deque<std::string> command;
};

}  // namespace ChessEngine

#endif  // PUMPKIN_UCI_POSITIONTASK_H
