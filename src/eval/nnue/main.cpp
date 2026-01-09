
#include <iostream>
#include <fstream>
#include <memory>

#include "nnue.h"

#include "../../game/Position.h"
#include "../evaluator.h"
#include "NnueEvaluator.h"

using namespace ChessEngine;
using namespace NNUE;

int main(int argc, char** argv) {
  std::shared_ptr<Nnue> nnue_model = std::make_shared<Nnue>();

  std::ifstream f("model.bin", std::ios::binary);
  nnue_model->load(f);
  nnue_model->randn_();
  std::shared_ptr<EvaluatorInterface> evaluator = std::make_shared<NnueEvaluator>(nnue_model);

  Position pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"); // 1. e4
  pos.set_listener(evaluator);

  evaluator->evaluate_black(pos);

  return 0;
}
