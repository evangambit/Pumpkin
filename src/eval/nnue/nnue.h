#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include <eigen3/Eigen/Dense>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "../../game/Position.h"
#include "utils.h"

namespace NNUE {

struct Nnue {
  bool x[INPUT_DIM];
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> embWeights[INPUT_DIM];
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> whiteAcc;
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> blackAcc;

  Eigen::Matrix<int16_t, Eigen::Dynamic, HIDDEN1_DIM> layer1;
  Eigen::Matrix<int16_t, 1, HIDDEN1_DIM> bias1;
  Eigen::Matrix<int32_t, 1, HIDDEN1_DIM> hidden1;

  Eigen::Matrix<int16_t, HIDDEN1_DIM, HIDDEN2_DIM> layer2;
  Eigen::Matrix<int16_t, 1, HIDDEN2_DIM> bias2;
  Eigen::Matrix<int32_t, 1, HIDDEN2_DIM> hidden2;

  Eigen::Matrix<int16_t, HIDDEN2_DIM, OUTPUT_DIM> layer3;
  Eigen::Matrix<int16_t, 1, OUTPUT_DIM> bias3;
  Eigen::Matrix<int16_t, 1, OUTPUT_DIM> output;

  Nnue() {
    std::fill_n(x, INPUT_DIM, false);
    whiteAcc.setZero();
    blackAcc.setZero();
    layer1.setZero(EMBEDDING_DIM, HIDDEN1_DIM);
    bias1.setZero();
    hidden1.setZero();
    layer2.setZero();
    bias2.setZero();
    hidden2.setZero();
    layer3.setZero();
    bias3.setZero();
    output.setZero();
  }

  void increment(size_t index) {
    assert(!x[index]);
    x[index] = true;
    whiteAcc += embWeights[index];
    blackAcc += embWeights[flip_feature_index(index)];
  }

  void decrement(size_t index) {
    assert(x[index]);
    x[index] = false;
    whiteAcc -= embWeights[index];
    blackAcc -= embWeights[flip_feature_index(index)];
  }

  void clear_accumulator() {
    whiteAcc.setZero();
    blackAcc.setZero();
  }

  void randn_() {
    /**
      Initialize weights and biases with gaussian random values.
     */
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      embWeights[i].array() = Eigen::Array<int16_t, 1, EMBEDDING_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / MAX_NUM_ONES_IN_INPUT)) * (1 << SCALE_SHIFT)); });
    }
    layer1.array() = Eigen::Array<int16_t, EMBEDDING_DIM, HIDDEN1_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / (EMBEDDING_DIM))) * (1 << SCALE_SHIFT)); });
    bias1.array() = Eigen::Array<int16_t, 1, HIDDEN1_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
    layer2.array() = Eigen::Array<int16_t, HIDDEN1_DIM, HIDDEN2_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / HIDDEN1_DIM)) * (1 << SCALE_SHIFT)); });
    bias2.array() = Eigen::Array<int16_t, 1, HIDDEN2_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
    layer3.array() = Eigen::Array<int16_t, HIDDEN2_DIM, OUTPUT_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(randn(std::sqrt(1.0 / HIDDEN2_DIM)) * (1 << SCALE_SHIFT)); });
    bias3.array() = Eigen::Array<int16_t, 1, OUTPUT_DIM>::Zero().unaryExpr([](int16_t) { return int16_t(0); });
  }

  void compute_acc_from_scratch(const ChessEngine::Position& pos) {
    std::fill_n(x, INPUT_DIM, false);
    whiteAcc.setZero();
    blackAcc.setZero();
    Features features = pos2features(pos);
    for (size_t i = 0; i < features.length; ++i) {
      size_t index = features[i];
      x[index] = true;
    }

    whiteAcc.setZero();
    blackAcc.setZero();
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      if (x[i]) {
        whiteAcc += embWeights[i];
        blackAcc += embWeights[flip_feature_index(i)];
      }
    }
  }

  static Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> load_matrix(std::istream& in) {
    char name[16];
    in.read(name, 16);
    std::string nameStr(name, 16);
    nameStr.erase(std::find_if(nameStr.rbegin(), nameStr.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), nameStr.end());

    // Read matrix dimensions
    uint32_t degree;
    in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
    uint32_t rows, cols;
    if (degree == 1) {
      rows = 1;
      in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    } else if (degree == 2) {
      in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
      in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    } else {
      throw std::runtime_error("Only 1D and 2D matrices are supported");
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(rows, cols);
    in.read(reinterpret_cast<char*>(mat.data()), sizeof(float) * rows * cols);

    size_t low = 0;
    size_t high = 0;
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        if (std::isnan(mat(i, j)) || std::isinf(mat(i, j))) {
          throw std::runtime_error("Matrix contains NaN or Inf values");
        }
        if (std::abs(mat(i, j)) > 1.0) {
          high++;
        } else {
          low++;
        }
      }
    }

    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> mat_int = mat.unaryExpr([](float v) {
      return static_cast<int16_t>(std::clamp(std::round(v * (1 << SCALE_SHIFT)), -32768.0f, 32767.0f));
    });

    return mat_int;
  }

  void load(std::istream& in) {
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> emb = load_matrix(in);;
    assert(emb.rows() == INPUT_DIM && emb.cols() == EMBEDDING_DIM);
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      embWeights[i] = emb.row(i);
    }

    layer1 = load_matrix(in).transpose();
    bias1 = load_matrix(in);
    layer2 = load_matrix(in).transpose();
    bias2 = load_matrix(in);
    layer3 = load_matrix(in).transpose();
    bias3 = load_matrix(in);

    // Verify that the entire file has been read
    char dummy;
    if (in.read(&dummy, 1) || !in.eof()) {
      throw std::runtime_error("File not completely read");
    }
  }

  int16_t *forward(ChessEngine::Color sideToMove) {
    if (sideToMove == ChessEngine::Color::WHITE) {
      hidden1.noalias() = whiteAcc.cast<int32_t>() * layer1.cast<int32_t>();
    } else {
      hidden1.noalias() = blackAcc.cast<int32_t>() * layer1.cast<int32_t>();
    }
    hidden1 += bias1.cast<int32_t>();
    
    // Debug: Print whiteAcc and hidden1 values
    
    // Right-shift by SCALE_SHIFT to account for fixed-point, then clip to [0, 64]
    hidden1.array() = (hidden1.array() / (1 << SCALE_SHIFT)).cwiseMax(0).cwiseMin(64);
    
    hidden2.noalias() = hidden1.cast<int32_t>() * layer2.cast<int32_t>();
    hidden2 += bias2.cast<int32_t>();
    
    hidden2.array() = (hidden2.array() / (1 << SCALE_SHIFT)).cwiseMax(0).cwiseMin(64);
    
    output.noalias() = (hidden2 * layer3.cast<int32_t>() + bias3.cast<int32_t>()).cast<int16_t>();
    
    return output.data();
  }

  std::shared_ptr<Nnue> clone() const {
    std::shared_ptr<Nnue> copy = std::make_shared<Nnue>();
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      copy->embWeights[i] = this->embWeights[i];
    }
    copy->layer1 = this->layer1;
    copy->bias1 = this->bias1;
    copy->layer2 = this->layer2;
    copy->bias2 = this->bias2;
    copy->layer3 = this->layer3;
    copy->bias3 = this->bias3;
    return copy;
  }
};

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUE_H