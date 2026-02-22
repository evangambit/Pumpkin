#include "../eval/nnue/NnueEvaluator.h"
#include "../game/Position.h"
#include <fstream>
#include <iostream>

using namespace NNUE;
using namespace ChessEngine;

int main() {
    auto model = std::make_shared<Nnue<float>>();
    std::ifstream in("hanging-1d.bin", std::ios::binary);
    model->load(in);
    
    // Position after e4
    Position pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    // manually print features
    NnueEvaluator<float> evaluator(model);
    Threats threats;
    create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
    Features features = pos2features(&evaluator, pos, threats);
    std::cout << "Features: ";
    for (size_t i = 0; i < features.length; ++i) {
        std::cout << features[i] << " ";
    }
    std::cout << "\n";
    model->compute_acc_from_scratch(features);
    
    std::cout << "White Acc: " << model->whiteAcc.data[0] << std::endl;
    std::cout << "Black Acc: " << model->blackAcc.data[0] << std::endl;
    std::cout << "layer1: " << model->layer1.data[0] << ", " << model->layer1.data[1] << std::endl;
    std::cout << "bias1: " << model->bias1.data[0] << std::endl;
    
    float* eval = model->forward(pos.turn_);
    std::cout << "Final eval: " << eval[0] << std::endl;
    return 0;
}
