#include <ATen/core/ivalue.h>
#include <torch/extension.h>
#include <fstream>
#include <string>
#include <vector>

#include "../src/game/Position.h"
#include "../src/game/Threats.h"
#include "../src/game/CreateThreats.h"
#include "../src/eval/nnue/Utils.h"

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
        std::vector<int16_t> all_values;
        std::vector<int16_t> all_lengths;
        std::vector<float> all_evals;
        
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
            if (pos.turn_ == Color::BLACK) {
                eval *= -1.0;
            }
            Threats threats;
            create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);

            NNUE::Features features = NNUE::pos2features<int>(pos, threats);
            
            for (size_t i = 0; i < features.length; i++) {
                all_values.push_back(features[i]);
            }
            all_lengths.push_back(features.length);
            all_evals.push_back(eval);
            lines_read++;
        }

        if (lines_read == 0) {
            throw pybind11::stop_iteration();
        }

        auto values_tensor = torch::empty({(long)all_values.size()}, torch::TensorOptions().dtype(torch::kInt16));
        std::memcpy(values_tensor.data_ptr<int16_t>(), all_values.data(), all_values.size() * sizeof(int16_t));
        
        auto lengths_tensor = torch::empty({(long)all_lengths.size()}, torch::TensorOptions().dtype(torch::kInt16));
        std::memcpy(lengths_tensor.data_ptr<int16_t>(), all_lengths.data(), all_lengths.size() * sizeof(int16_t));

        auto evals_tensor = torch::empty({(long)all_evals.size()}, torch::TensorOptions().dtype(torch::kFloat32));
        std::memcpy(evals_tensor.data_ptr<float>(), all_evals.data(), all_evals.size() * sizeof(float));

        return {values_tensor, lengths_tensor, evals_tensor};
    }
};

PYBIND11_MODULE(_nnue_dataset, m) {
    pybind11::class_<ChunkedDataset>(m, "ChunkedDataset")
        .def(pybind11::init<std::vector<std::string>, int>())
        .def("__iter__", [](ChunkedDataset& s) -> ChunkedDataset& { return s; })
        .def("__next__", &ChunkedDataset::next);
}
