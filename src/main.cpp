#include "search/search.h"
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
    auto evaluator = std::make_shared<SimpleEvaluator>();

    Thread thread(0, pos, evaluator, std::unordered_set<Move>());
    SearchResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(&thread, 4, ColoredEvaluation<Color::WHITE>(kMinEval), ColoredEvaluation<Color::WHITE>(kMaxEval));

    std::cout << "Best Move: " << result.bestMove.uci() << ", Evaluation: " << result.evaluation << std::endl;

    return 0;
}