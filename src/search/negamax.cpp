#include "negamax.h"

#include <iostream>
#include <cmath>

namespace ChessEngine {

#if COLLECT_MOVE_ORDER_STATISTICS

static MoveOrderStatistics moveOrderStatistics_;

double logit(double p) {
  return std::log(p / (1.0 - p));
}

void collect_move_order_statistics(Thread* thread) {
  moveOrderStatistics_ += thread->moveOrderStatistics_;
}

void print_move_order_statistics() {
  const uint64_t bestMoveCount = moveOrderStatistics_.bestMoveCount_;
  const uint64_t moveCount = moveOrderStatistics_.moveCount_;
  const uint64_t notBestMoveCount = moveCount - bestMoveCount;

  std::cout << "Best move count: " << bestMoveCount << std::endl;
  std::cout << "Move count: " << moveCount << std::endl;

  // std::vector<std::string> keys;
  // double avg = 0.0;
  // for (const auto& entry : moveOrderStatistics_.featureCounts_) {
  //   keys.push_back(entry.first);
  //   avg += moveOrderStatistics_.f(entry.first).first;
  // }
  // std::sort(keys.begin(), keys.end());
  // avg /= keys.size();

  int16_t *scores = new int16_t[7 * 7 * 2 * 2];
  std::fill(scores, scores + 7 * 7 * 2 * 2, 0);
  for (const auto& entry : moveOrderStatistics_.featureCounts_) {
    try {
      unsigned key = std::stoul(entry.first);
      if (key >= 7 * 7 * 2 * 2) {
        std::cout << "oops " << entry.first << std::endl;
        continue;
      }
      int16_t score = moveOrderStatistics_.f(entry.first).first * 100;
      scores[key] = score;
    } catch (std::invalid_argument&) {
      auto r = moveOrderStatistics_.f(entry.first);
      const double weight = r.first;
      const double ci95 = r.second;
      std::cout << entry.first << ": " << weight << " (95% CI: [" << weight - ci95 << ", " << weight + ci95 << "])" << std::endl;
    }
  }

  char names[8] = "?PNBRQK";
  for (unsigned i = 0; i < 7 * 7 * 2 * 2; ++i) {
    std::cout << scores[i] << ", ";
    if (i % 4 == 3) {
      std::cout << " // " << names[i / 28] << "x" << names[(i / 4) % 7] << std::endl;
    }
  }

  // for (const auto& key : keys) {
  //   auto r = moveOrderStatistics_.f(key);
  //   const double weight = r.first - avg;
  //   const double ci95 = r.second;
  //   std::cout << key << ": " << weight << " (95% CI: [" << weight - ci95 << ", " << weight + ci95 << "])" << std::endl;
  // }
}

#endif

}  // namespace ChessEngine