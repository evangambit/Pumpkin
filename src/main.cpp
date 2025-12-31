
#include <gflags/gflags.h>
#include "search/search.h"
#include "search/evaluator.h"
#include "search/PieceSquareEvaluator.h"
#include "game/Position.h"
#include "string_utils.h"


DEFINE_string(fen, "", "FEN string for the chess position");
DEFINE_int32(depth, 5, "Search depth");

using namespace ChessEngine;

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_fen.empty()) {
        std::cerr << "Error: --fen flag is required." << std::endl;
        return 1;
    }

    Position pos(FLAGS_fen);
    auto evaluator = std::make_shared<PieceSquareEvaluator>();
    std::pair<ColoredEvaluation<Color::WHITE>, Move> result = search(pos, evaluator, FLAGS_depth);

    std::cout << "Best Move: " << result.second.uci() << ", Evaluation: " << result.first << std::endl;

    return 0;
}