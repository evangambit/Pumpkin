#include "search/search.h"
#include "search/evaluator.h"
#include "search/PieceSquareEvaluator.h"
#include "game/Position.h"
#include "string_utils.h"

using namespace ChessEngine;

int main(int argc, char** argv) {

    std::vector<std::string> fenParts;
    for (int i = 1; i < argc; ++i) {
        fenParts.push_back(argv[i]);
    }
    std::string fen = join(fenParts, " ");

    Position pos(fen);
    auto evaluator = std::make_shared<PieceSquareEvaluator>();
    std::pair<ColoredEvaluation<Color::WHITE>, Move> result = search(pos, evaluator, 4);

    std::cout << "Best Move: " << result.second.uci() << ", Evaluation: " << result.first << std::endl;

    return 0;
}