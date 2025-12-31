
#include <gflags/gflags.h>
#include "search/search.h"
#include "search/evaluator.h"
#include "search/PieceSquareEvaluator.h"
#include "game/Position.h"
#include "string_utils.h"


DEFINE_string(fen, "", "FEN string for the chess position");
DEFINE_int32(depth, 5, "Search depth");
DEFINE_int32(multi_pv, 1, "Number of principal variations to search for");

using namespace ChessEngine;

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    initialize_geometry();
    initialize_zorbrist();
    initialize_movegen();

    if (FLAGS_fen.empty()) {
        std::cerr << "Error: --fen flag is required." << std::endl;
        return 1;
    }

    Position pos(FLAGS_fen);
    auto evaluator = std::make_shared<PieceSquareEvaluator>();
    auto result = search(pos, evaluator, FLAGS_depth, FLAGS_multi_pv);

    std::cout << "Best Move: " << result.bestMove.uci() << ", Evaluation: " << result.evaluation << std::endl;
    for (const auto& v : result.primaryVariations) {
        std::cout << "PV Move: " << v.first.uci() << ", Eval: " << v.second << std::endl;
    }

    return 0;
}