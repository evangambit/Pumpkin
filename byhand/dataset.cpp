#include <ATen/core/ivalue.h>
#include <torch/extension.h>
#include <fstream>
#include <string>
#include <vector>

#include "../src/game/Position.h"
#include "../src/game/Threats.h"
#include "../src/game/CreateThreats.h"
#include "../src/game/movegen/movegen.h"
#include "../src/eval/byhand/byhand.h"

using namespace ChessEngine;

struct ChunkedDataset {
    std::vector<std::string> paths;
    int chunk_size;
    
    size_t file_idx = 0;
    std::ifstream current_file;

    ChunkedDataset(std::vector<std::string> paths, int chunk_size)
        : paths(paths), chunk_size(chunk_size), file_idx(0) {
        if (file_idx < this->paths.size()) {
            current_file.open(this->paths[file_idx]);
        }
    }

    std::vector<torch::Tensor> next() {
        std::vector<int8_t> all_values;
        std::vector<float> all_evals;
        std::vector<int8_t> all_turns;
        std::vector<int16_t> pst_values;
        std::vector<int16_t> pst_signs;
        std::vector<int16_t> pst_lengths;
        
        int lines_read = 0;
        std::string line;

        while (lines_read < chunk_size) {
            if (!current_file.is_open() || !std::getline(current_file, line)) {
                if (current_file.is_open()) {
                    current_file.close();
                }
                file_idx++;
                if (file_idx >= paths.size()) {
                    break;
                }
                current_file.open(paths[file_idx]);
                continue;
            }

            size_t bar_pos = line.find('|');
            if (bar_pos == std::string::npos) continue;

            std::string fen = line.substr(0, bar_pos);
            std::string eval_str = line.substr(bar_pos + 1);
            float eval = std::stof(eval_str);

            Position pos(fen);
            Threats threats;
            create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);

            int8_t features[ByHand::EF_COUNT] = {0};
            if (pos.turn_ == Color::WHITE) {
                ByHand::pos2features<Color::WHITE>(pos, threats, features);
            } else {
                ByHand::pos2features<Color::BLACK>(pos, threats, features);
            }
            
            for (size_t i = 0; i < ByHand::EF_COUNT; i++) {
                all_values.push_back(features[i]);
            }
            all_evals.push_back(eval);
            all_turns.push_back(static_cast<int8_t>(pos.turn_));

            const int whiteOffset = pos.turn_ == Color::WHITE ? 0 : 6;
            const int blackOffset = pos.turn_ == Color::BLACK ? 0 : 6;
            size_t bag_start = pst_values.size();
            for (Piece i = Piece::PAWN; i <= Piece::KING; i = Piece(i + 1)) {
                const Bitboard white = pos.pieceBitboards_[coloredPiece(Color::WHITE, i)];
                const Bitboard black = pos.pieceBitboards_[coloredPiece(Color::BLACK, i)];
                // We can save a little bit of compute while training by dropping pieces
                // that have corresponding enemies on the board (e.g. a white knight on F3
                // and a black knight on F6). To do this symmetrically, we vertical-flip black.
                const Bitboard flippedBlack = flip_vertically(black);
                const Bitboard overlap = white & flippedBlack;
                Bitboard onlyWhite = white ^ overlap;
                Bitboard onlyBlackFlipped = flippedBlack ^ overlap;

                while (onlyWhite) {
                    int sq = pop_lsb_i_promise_board_is_not_empty(onlyWhite);
                    pst_values.push_back((i - Piece::PAWN + whiteOffset) * 64 + sq);
                }
                while (onlyBlackFlipped) {
                    int sq = pop_lsb_i_promise_board_is_not_empty(onlyBlackFlipped);
                    pst_values.push_back((i - Piece::PAWN + blackOffset) * 64 + sq);
                }
            }
            pst_lengths.push_back(pst_values.size() - bag_start);

            lines_read++;
        }

        if (lines_read == 0) {
            throw pybind11::stop_iteration();
        }

        auto values_tensor = torch::empty({lines_read, ByHand::EF_COUNT}, torch::TensorOptions().dtype(torch::kInt8));
        std::memcpy(values_tensor.data_ptr<int8_t>(), all_values.data(), all_values.size() * sizeof(int8_t));
        
        auto evals_tensor = torch::empty({(long)all_evals.size()}, torch::TensorOptions().dtype(torch::kFloat32));
        std::memcpy(evals_tensor.data_ptr<float>(), all_evals.data(), all_evals.size() * sizeof(float));

        auto turn_tensor = torch::empty({(long)all_turns.size()}, torch::TensorOptions().dtype(torch::kInt8));
        std::memcpy(turn_tensor.data_ptr<int8_t>(), all_turns.data(), all_turns.size() * sizeof(int8_t));

        auto pst_values_tensor = torch::empty({(long)pst_values.size()}, torch::TensorOptions().dtype(torch::kInt16));
        std::memcpy(pst_values_tensor.data_ptr<int16_t>(), pst_values.data(), pst_values.size() * sizeof(int16_t));
        auto pst_lengths_tensor = torch::empty({(long)pst_lengths.size()}, torch::TensorOptions().dtype(torch::kInt16));
        std::memcpy(pst_lengths_tensor.data_ptr<int16_t>(), pst_lengths.data(), pst_lengths.size() * sizeof(int16_t));

        return {values_tensor, evals_tensor, turn_tensor, pst_values_tensor, pst_lengths_tensor};
    }
};

PYBIND11_MODULE(_byhand_dataset, m) {
    pybind11::class_<ChunkedDataset>(m, "ChunkedDataset")
        .def(pybind11::init<std::vector<std::string>, int>())
        .def("__iter__", [](ChunkedDataset& s) -> ChunkedDataset& { return s; })
        .def("__next__", &ChunkedDataset::next);

    m.def("feature_name", [](int index) -> std::string {
        if (index < 0 || index >= ByHand::EF_COUNT) {
            throw pybind11::value_error("Feature index out of range");
        }
        return ByHand::to_string(static_cast<ByHand::EF>(index));
    }, "Returns the name of a feature given its index.");
    m.def("max_earliness", []() -> int {
        return ChessEngine::ByHand::kMaxEarliness;
    }, "Returns the maximum earliness value for a given feature index.");
    m.def("earliness_index", []() -> int {
        return ChessEngine::ByHand::EF::EARLINESS;
    }, "Returns the index of the earliness feature.");
}
