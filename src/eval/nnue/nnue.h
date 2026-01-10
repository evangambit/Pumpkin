#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "../../game/Position.h"
#include "utils.h"

namespace NNUE {

template <size_t HEIGHT, size_t WIDTH>
struct Matrix {
  int16_t *data;

  Matrix() {
    data = new int16_t[HEIGHT * WIDTH];
    setZero();
  }

  ~Matrix() {
    delete[] data;
  }

  Matrix& operator=(const Matrix& other) {
    std::memcpy(data, other.data, HEIGHT * WIDTH * sizeof(int16_t));
    return *this;
  }

  Matrix(const Matrix& other) {
    data = new int16_t[HEIGHT * WIDTH];
    std::memcpy(data, other.data, HEIGHT * WIDTH * sizeof(int16_t));
  }

  Matrix& operator=(Matrix&& other) noexcept {
    std::swap(data, other.data);
    return *this;
  }

  Matrix(Matrix&& other) noexcept : data(other.data) {
    other.data = nullptr;
  }

  void setZero() {
    std::fill(data, data + HEIGHT * WIDTH, int16_t(0));
  }

  void load_from_stream(std::istream& in) {
    char name[16];
    in.read(name, 16);

    uint32_t degree;
    in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
    if (degree != 2) {
      throw std::runtime_error("Only 2D matrices are supported");
    }
    uint32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    if (rows != HEIGHT || cols != WIDTH) {
      throw std::runtime_error("Matrix size mismatch");
    }
    float *buffer = new float[rows * cols];
    in.read(reinterpret_cast<char*>(buffer), sizeof(float) * rows * cols);
    for (size_t i = 0; i < HEIGHT; ++i) {
      for (size_t j = 0; j < WIDTH; ++j) {
        data[i * WIDTH + j] = static_cast<int16_t>(buffer[i * WIDTH + j] * (1 << SCALE_SHIFT));
      }
    }
    delete[] buffer;
  }

  friend std::ostream& operator<<(std::ostream& os, const Matrix<HEIGHT, WIDTH>& mat) {
    for (size_t i = 0; i < std::min(HEIGHT, size_t(4)); ++i) {
      for (size_t j = 0; j < std::min(WIDTH, size_t(4)); ++j) {
        os << mat.data[i * WIDTH + j] << " ";
      }
      if (WIDTH > 4) {
        os << "...";
      }
      os << std::endl;
    }
    return os;
  }

  void randn_() {
    for (size_t i = 0; i < HEIGHT; ++i) {
      for (size_t j = 0; j < WIDTH; ++j) {
        data[i * WIDTH + j] = static_cast<int16_t>(randn());
      }
    }
  }
};

template<size_t DIM>
struct Vector {
  int16_t *data;

  Vector() {
    data = new int16_t[DIM];
    setZero();
  }

  ~Vector() {
    delete[] data;
  }

  Vector& operator=(const Vector& other) {
    std::memcpy(data, other.data, DIM * sizeof(int16_t));
    return *this;
  }

  Vector(const Vector& other) {
    data = new int16_t[DIM];
    std::memcpy(data, other.data, DIM * sizeof(int16_t));
  }

  Vector& operator=(Vector&& other) noexcept {
    std::swap(data, other.data);
    return *this;
  }

  Vector(Vector&& other) noexcept : data(other.data) {
    other.data = nullptr;
  }

  void setZero() {
    std::fill(data, data + DIM, int16_t(0));
  }

  template<size_t HEIGHT>
  void load_from_row(const Matrix<HEIGHT, DIM>& mat, size_t rowIndex) {
    if (rowIndex >= HEIGHT) {
      throw std::runtime_error("Row index out of bounds");
    }
    for (size_t j = 0; j < DIM; ++j) {
      data[j] = mat.data[rowIndex * DIM + j];
    }
  }

  void load_from_stream(std::istream& in) {
    char name[16];
    in.read(name, 16);

    uint32_t degree;
    in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));

    uint32_t size;
    if (degree == 1) {
      in.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    } else if (degree == 2) {
      uint32_t rows, cols;
      in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
      in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
      if (rows != 1 || cols != 1) {
        throw std::runtime_error("Only vectors with one row are supported");
      }
      size = cols == 1 ? rows : cols;
    } else {
      throw std::runtime_error("Only 1D or 2D vectors are supported");
    }
    if (size != DIM) {
      throw std::runtime_error("Vector size mismatch");
    }

    float buffer[DIM];
    in.read(reinterpret_cast<char*>(buffer), sizeof(float) * size);
    for (size_t i = 0; i < DIM; ++i) {
      data[i] = static_cast<int16_t>(buffer[i] * (1 << SCALE_SHIFT));
    }
  }

  Vector<DIM>& operator+=(const Vector< DIM >& other) {
    for (size_t i = 0; i < DIM; ++i) {
      data[i] += other.data[i];
    }
    return *this;
  }

  Vector<DIM>& operator-=(const Vector< DIM >& other) {
    for (size_t i = 0; i < DIM; ++i) {
      data[i] -= other.data[i];
    }
    return *this;
  }

  void clip_(int16_t minVal, int16_t maxVal) {
    for (size_t i = 0; i < DIM; ++i) {
      data[i] = std::max(minVal, std::min(data[i], maxVal));
    }
  }

  void randn_() {
    for (size_t i = 0; i < DIM; ++i) {
      data[i] = static_cast<int16_t>(randn());
    }
  }

  int16_t* data_ptr() {
    return &data[0];
  }

  friend std::ostream& operator<<(std::ostream& os, const Vector<DIM>& vec) {
    for (size_t i = 0; i < std::min(DIM, size_t(10)); ++i) {
      os << vec.data[i] << " ";
    }
    if (DIM > 10) {
      os << "...";
    }
    os << std::endl;
    return os;
  }
};

template<size_t HEIGHT, size_t WIDTH>
inline void matmul(Matrix<HEIGHT, WIDTH>& mat, const Vector<WIDTH>& vec, Vector<HEIGHT>* out) {
  for (size_t i = 0; i < HEIGHT; ++i) {
    int32_t sum = 0;
    for (size_t j = 0; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
    out->data[i] = static_cast<int16_t>(sum >> SCALE_SHIFT);
  }
}

struct Nnue {
      // layer1.matmul(sideToMove == ChessEngine::Color::WHITE ? whiteAcc : blackAcc, &hidden1);

  bool x[INPUT_DIM];
  Vector<EMBEDDING_DIM> embWeights[INPUT_DIM];
  Vector<EMBEDDING_DIM> whiteAcc;
  Vector<EMBEDDING_DIM> blackAcc;

  Matrix<HIDDEN1_DIM, EMBEDDING_DIM> layer1;
  Vector<HIDDEN1_DIM> bias1;
  Vector<HIDDEN1_DIM> hidden1;

  Matrix<HIDDEN2_DIM, HIDDEN1_DIM> layer2;
  Vector<HIDDEN2_DIM> bias2;
  Vector<HIDDEN2_DIM> hidden2;
  Matrix<OUTPUT_DIM, HIDDEN2_DIM> layer3;
  Vector<OUTPUT_DIM> bias3;
  Vector<OUTPUT_DIM> output;

  Nnue() {
    std::fill_n(x, INPUT_DIM, false);
    whiteAcc.setZero();
    blackAcc.setZero();
    layer1.setZero();
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
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      embWeights[i].randn_();
    }
    layer1.randn_();
    bias1.randn_();
    layer2.randn_();
    bias2.randn_();
    layer3.randn_();
    bias3.randn_();
  }

  void compute_acc_from_scratch(const ChessEngine::Position& pos) {
    std::fill_n(x, INPUT_DIM, false);
    whiteAcc.setZero();
    blackAcc.setZero();
    Features features = pos2features(pos);
    for (size_t i = 0; i < features.length; ++i) {
      size_t index = features[i];
      x[index] = true;
      whiteAcc += embWeights[index];
      blackAcc += embWeights[flip_feature_index(index)];
    }
  }

  void load(std::istream& in) {
    Matrix<INPUT_DIM, EMBEDDING_DIM> emb;
    emb.load_from_stream(in);
    for (size_t i = 0; i < INPUT_DIM; ++i) {
      embWeights[i].load_from_row(emb, i);
    }
    layer1.load_from_stream(in);
    bias1.load_from_stream(in);
    layer2.load_from_stream(in);
    bias2.load_from_stream(in);
    layer3.load_from_stream(in);
    bias3.load_from_stream(in);

    // Verify that the entire file has been read
    char dummy;
    if (in.read(&dummy, 1) || !in.eof()) {
      throw std::runtime_error("File not completely read");
    }
  }

  int16_t *forward(ChessEngine::Color sideToMove) {
    matmul(layer1, sideToMove == ChessEngine::Color::WHITE ? whiteAcc : blackAcc, &hidden1);
    hidden1 += bias1;
    hidden1.clip_(0, 64);

    matmul(layer2, hidden1, &hidden2);
    hidden2 += bias2;
    hidden2.clip_(0, 64);

    matmul(layer3, hidden2, &output);
    output += bias3;

    return output.data_ptr();
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