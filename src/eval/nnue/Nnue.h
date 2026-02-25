#ifndef SRC_EVAL_NNUE_NNUE_H
#define SRC_EVAL_NNUE_NNUE_H

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "../../game/Position.h"
#include "Utils.h"

namespace NNUE {

template <size_t HEIGHT, size_t WIDTH, typename T>
struct Matrix {
  static_assert(WIDTH > 0 && HEIGHT > 0, "Matrix dimensions must be greater than zero");
  static_assert(std::is_same<T, int16_t>::value || std::is_same<T, int8_t>::value || std::is_same<T, float>::value, "Matrix type must be int16_t, int8_t, or float");
  alignas(16) T data[HEIGHT * WIDTH];

  Matrix() {
    setZero();
  }

  ~Matrix() = default;

  Matrix& operator=(const Matrix& other) = default;

  Matrix(const Matrix& other) = default;

  Matrix& operator=(Matrix&& other) = default;

  Matrix(Matrix&& other) = default;

  void setZero() {
    std::fill(data, data + HEIGHT * WIDTH, T(0));
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
      throw std::runtime_error("Matrix size mismatch; expected " + std::to_string(HEIGHT) + "x" + std::to_string(WIDTH) + ", got " + std::to_string(rows) + "x" + std::to_string(cols));
    }
    float *buffer = new float[rows * cols];
    in.read(reinterpret_cast<char*>(buffer), sizeof(float) * rows * cols);
    for (size_t i = 0; i < HEIGHT; ++i) {
      for (size_t j = 0; j < WIDTH; ++j) {
        if (std::is_same<T, float>::value) {
          data[i * WIDTH + j] = buffer[i * WIDTH + j];
        } else if (std::is_same<T, int16_t>::value) {
          data[i * WIDTH + j] = static_cast<T>(buffer[i * WIDTH + j] * (1 << SCALE_SHIFT));
        } else if (std::is_same<T, int8_t>::value) {
          // Quantize weights down to fit perfectly inside an int8_t
          // SCALE_SHIFT is 8, meaning we scale floats by 256. We need to scale by 64 instead (SCALE_SHIFT - 2)
          data[i * WIDTH + j] = static_cast<T>(buffer[i * WIDTH + j] * (1 << (SCALE_SHIFT - 2)));
        }
      }
    }
    delete[] buffer;
  }

  int16_t max() const {
    int16_t maxVal = data[0];
    for (size_t i = 1; i < HEIGHT * WIDTH; ++i) {
      if (data[i] > maxVal) {
        maxVal = data[i];
      }
    }
    return maxVal;
  }

  int16_t min() const {
    int16_t minVal = data[0];
    for (size_t i = 1; i < HEIGHT * WIDTH; ++i) {
      if (data[i] < minVal) {
        minVal = data[i];
      }
    }
    return minVal;
  }

  void randn_() {
    for (size_t i = 0; i < HEIGHT; ++i) {
      for (size_t j = 0; j < WIDTH; ++j) {
        data[i * WIDTH + j] = static_cast<T>(randn());
      }
    }
  }

  bool operator==(const Matrix<HEIGHT, WIDTH, T>& other) const {
    for (size_t i = 0; i < HEIGHT * WIDTH; ++i) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(int16_t val) const {
    for (size_t i = 0; i < HEIGHT * WIDTH; ++i) {
      if (data[i] != val) {
        return false;
      }
    }
    return true;
  }
};

template<size_t HEIGHT, size_t WIDTH, typename T>
inline std::ostream& operator<<(std::ostream& os, const Matrix<HEIGHT, WIDTH, T>& mat) {
  for (size_t i = 0; i < std::min(HEIGHT, size_t(5)); ++i) {
    for (size_t j = 0; j < std::min(WIDTH, size_t(5)); ++j) {
      os << mat.data[i * WIDTH + j] << " ";
    }
    if (WIDTH > 5) {
      os << "...";
    }
    os << std::endl;
  }
  if (HEIGHT > 5) {
    os << "...";
  }
  std::cout << "( max: " << mat.max() << ", min: " << mat.min() << " )" << std::endl;
  return os;
}

template<size_t DIM, typename T>
struct Vector {
  static_assert(DIM > 0, "Vector dimension must be greater than zero");
  static_assert(std::is_same<T, int16_t>::value || std::is_same<T, int8_t>::value || std::is_same<T, float>::value, "Vector type must be int16_t, int8_t, or float");
  alignas(16) T data[DIM];

  T operator[](size_t index) const {
    return data[index];
  }

  Vector() {
    setZero();
  }

  ~Vector() = default;

  Vector& operator=(const Vector& other) = default;

  Vector(const Vector& other) = default;

  Vector& operator=(Vector&& other) = default;

  Vector(Vector&& other) = default;

  void setZero() {
    std::fill(data, data + DIM, T(0));
  }

  template<size_t HEIGHT>
  void load_from_row(const Matrix<HEIGHT, DIM, T>& mat, size_t rowIndex) {
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
      if (std::is_same<T, float>::value) {
        data[i] = buffer[i];
      } else if (std::is_same<T, int16_t>::value) {
        data[i] = static_cast<T>(buffer[i] * (1 << SCALE_SHIFT));
      } else if (std::is_same<T, int8_t>::value) {
        data[i] = static_cast<T>(buffer[i] * (1 << (SCALE_SHIFT - 2)));
      }
    }
  }

  Vector<DIM, T>& operator+=(const Vector< DIM, T >& other) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    if constexpr (std::is_same<T, int16_t>::value) {
      for (size_t i = 0; i < DIM; i += 8) {
        int16x8_t a = vld1q_s16(&data[i]);
        int16x8_t b = vld1q_s16(&other.data[i]);
        vst1q_s16(&data[i], vaddq_s16(a, b));
      }
      return *this;
    }
#endif
    for (size_t i = 0; i < DIM; ++i) {
      data[i] += other.data[i];
    }
    return *this;
  }

  Vector<DIM, T>& operator-=(const Vector<DIM, T>& other) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    if constexpr (std::is_same<T, int16_t>::value) {
      for (size_t i = 0; i < DIM; i += 8) {
        int16x8_t a = vld1q_s16(&data[i]);
        int16x8_t b = vld1q_s16(&other.data[i]);
        vst1q_s16(&data[i], vsubq_s16(a, b));
      }
      return *this;
    }
#endif
    for (size_t i = 0; i < DIM; ++i) {
      data[i] -= other.data[i];
    }
    return *this;
  }

  void clip_(T minVal, T maxVal) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    if constexpr (std::is_same<T, int16_t>::value) {
      int16x8_t vMin = vdupq_n_s16(minVal);
      int16x8_t vMax = vdupq_n_s16(maxVal);
      for (size_t i = 0; i < DIM; i += 8) {
        int16x8_t v = vld1q_s16(&data[i]);
        v = vmaxq_s16(vMin, vminq_s16(v, vMax));
        vst1q_s16(&data[i], v);
      }
      return;
    }
#endif
    for (size_t i = 0; i < DIM; ++i) {
      data[i] = std::max<T>(minVal, std::min<T>(data[i], maxVal));
    }
  }

  void randn_() {
    for (size_t i = 0; i < DIM; ++i) {
      data[i] = static_cast<T>(randn());
    }
  }

  T* data_ptr() {
    return &data[0];
  }

  bool operator==(const Vector<DIM, T>& other) const {
    for (size_t i = 0; i < DIM; ++i) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(int16_t val) const {
    for (size_t i = 0; i < DIM; ++i) {
      if (data[i] != val) {
        return false;
      }
    }
    return true;
  }

  bool isZero() const {
    for (size_t i = 0; i < DIM; ++i) {
      if (data[i] != 0) {
        return false;
      }
    }
    return true;
  }

  Vector<DIM, T> operator-(const Vector<DIM, T>& other) const {
    Vector<DIM, T> result;
    for (size_t i = 0; i < DIM; ++i) {
      result.data[i] = this->data[i] - other.data[i];
    }
    return result;
  }

  void print_diff(const Vector<DIM, T>& other) const {
    for (size_t i = 0; i < DIM; ++i) {
      if (this->data[i] != other.data[i]) {
        std::cout << "Index " << i << ": " << this->data[i] << " vs " << other.data[i] << std::endl;
      }
    }
  }
};

template<size_t DIM, typename T>
inline std::ostream& operator<<(std::ostream& os, const Vector<DIM, T>& vec) {
  for (size_t i = 0; i < std::min(DIM, size_t(10)); ++i) {
    os << vec.data[i] << " ";
  }
  if (DIM > 10) {
    os << "...";
  }
  os << std::endl;
  return os;
}

template<size_t HEIGHT, size_t WIDTH>
inline void matmul(Matrix<HEIGHT, WIDTH, float>& mat, const Vector<WIDTH, float>& vec, Vector<HEIGHT, float>* out) {
  for (size_t i = 0; i < HEIGHT; ++i) {
    float sum = 0;
    for (size_t j = 0; j < WIDTH; ++j) {
      sum += mat.data[i * WIDTH + j] * vec.data[j];
    }
    out->data[i] = sum;
  }
}

template<size_t HEIGHT, size_t WIDTH>
inline void matmul(Matrix<HEIGHT, WIDTH, int16_t>& mat, const Vector<WIDTH, int16_t>& vec, Vector<HEIGHT, int16_t>* out) {
  for (size_t i = 0; i < HEIGHT; ++i) {
    int32_t sum = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    for (size_t j = 0; j < WIDTH; j += 32) {
      int16x8_t m0 = vld1q_s16(&mat.data[i * WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec.data[j]);
      int16x8_t m1 = vld1q_s16(&mat.data[i * WIDTH + j + 8]);
      int16x8_t v1 = vld1q_s16(&vec.data[j + 8]);
      int16x8_t m2 = vld1q_s16(&mat.data[i * WIDTH + j + 16]);
      int16x8_t v2 = vld1q_s16(&vec.data[j + 16]);
      int16x8_t m3 = vld1q_s16(&mat.data[i * WIDTH + j + 24]);
      int16x8_t v3 = vld1q_s16(&vec.data[j + 24]);

      sum0 = vmlal_s16(sum0, vget_low_s16(m0), vget_low_s16(v0));
      sum0 = vmlal_s16(sum0, vget_high_s16(m0), vget_high_s16(v0));
      
      sum1 = vmlal_s16(sum1, vget_low_s16(m1), vget_low_s16(v1));
      sum1 = vmlal_s16(sum1, vget_high_s16(m1), vget_high_s16(v1));

      sum2 = vmlal_s16(sum2, vget_low_s16(m2), vget_low_s16(v2));
      sum2 = vmlal_s16(sum2, vget_high_s16(m2), vget_high_s16(v2));

      sum3 = vmlal_s16(sum3, vget_low_s16(m3), vget_low_s16(v3));
      sum3 = vmlal_s16(sum3, vget_high_s16(m3), vget_high_s16(v3));
    }
    int32x4_t sum_vec = vaddq_s32(vaddq_s32(sum0, sum1), vaddq_s32(sum2, sum3));
    sum = vaddvq_s32(sum_vec);
#else
    for (size_t j = 0; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#endif
    sum >>= SCALE_SHIFT;
    out->data[i] = static_cast<int16_t>(std::max(-(1 << 15), std::min(static_cast<int32_t>(1 << 15) - 1, sum)));
  }
}

/**
 * Performs matmul(mat, concat(vec1, vec2)), where concat(vec1, vec2) is the concatenation of the two vectors.
 */
template<size_t HEIGHT, size_t COMBINED_WIDTH, size_t WIDTH1, size_t WIDTH2>
inline void concat_and_matmul(const Matrix<HEIGHT, COMBINED_WIDTH, int16_t>& mat, const Vector<WIDTH1, int16_t>& vec1, const Vector<WIDTH2, int16_t>& vec2, Vector<HEIGHT, int16_t>* out) {
  static_assert(WIDTH1 + WIDTH2 == COMBINED_WIDTH, "Matrix width must match the sum of the two vector widths");
  for (size_t i = 0; i < HEIGHT; ++i) {
    int32_t sum = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    for (size_t j = 0; j < WIDTH1; j += 32) {
      int16x8_t m0 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec1.data[j]);
      int16x8_t m1 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + j + 8]);
      int16x8_t v1 = vld1q_s16(&vec1.data[j + 8]);
      int16x8_t m2 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + j + 16]);
      int16x8_t v2 = vld1q_s16(&vec1.data[j + 16]);
      int16x8_t m3 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + j + 24]);
      int16x8_t v3 = vld1q_s16(&vec1.data[j + 24]);

      sum0 = vmlal_s16(sum0, vget_low_s16(m0), vget_low_s16(v0));
      sum0 = vmlal_s16(sum0, vget_high_s16(m0), vget_high_s16(v0));
      sum1 = vmlal_s16(sum1, vget_low_s16(m1), vget_low_s16(v1));
      sum1 = vmlal_s16(sum1, vget_high_s16(m1), vget_high_s16(v1));
      sum2 = vmlal_s16(sum2, vget_low_s16(m2), vget_low_s16(v2));
      sum2 = vmlal_s16(sum2, vget_high_s16(m2), vget_high_s16(v2));
      sum3 = vmlal_s16(sum3, vget_low_s16(m3), vget_low_s16(v3));
      sum3 = vmlal_s16(sum3, vget_high_s16(m3), vget_high_s16(v3));
    }
    for (size_t j = 0; j < WIDTH2; j += 32) {
      int16x8_t m0 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j]);
      int16x8_t v0 = vld1q_s16(&vec2.data[j]);
      int16x8_t m1 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 8]);
      int16x8_t v1 = vld1q_s16(&vec2.data[j + 8]);
      int16x8_t m2 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 16]);
      int16x8_t v2 = vld1q_s16(&vec2.data[j + 16]);
      int16x8_t m3 = vld1q_s16(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 24]);
      int16x8_t v3 = vld1q_s16(&vec2.data[j + 24]);

      sum0 = vmlal_s16(sum0, vget_low_s16(m0), vget_low_s16(v0));
      sum0 = vmlal_s16(sum0, vget_high_s16(m0), vget_high_s16(v0));
      sum1 = vmlal_s16(sum1, vget_low_s16(m1), vget_low_s16(v1));
      sum1 = vmlal_s16(sum1, vget_high_s16(m1), vget_high_s16(v1));
      sum2 = vmlal_s16(sum2, vget_low_s16(m2), vget_low_s16(v2));
      sum2 = vmlal_s16(sum2, vget_high_s16(m2), vget_high_s16(v2));
      sum3 = vmlal_s16(sum3, vget_low_s16(m3), vget_low_s16(v3));
      sum3 = vmlal_s16(sum3, vget_high_s16(m3), vget_high_s16(v3));
    }
    int32x4_t sum_vec = vaddq_s32(vaddq_s32(sum0, sum1), vaddq_s32(sum2, sum3));
    sum = vaddvq_s32(sum_vec);
#else
    for (size_t j = 0; j < WIDTH1; ++j) {
      sum += static_cast<int32_t>(mat.data[i * COMBINED_WIDTH + j]) * static_cast<int32_t>(vec1.data[j]);
    }
    for (size_t j = 0; j < WIDTH2; ++j) {
      sum += static_cast<int32_t>(mat.data[i * COMBINED_WIDTH + WIDTH1 + j]) * static_cast<int32_t>(vec2.data[j]);
    }
#endif
    sum >>= SCALE_SHIFT;
    out->data[i] = static_cast<int16_t>(std::max(-(1 << 15), std::min(static_cast<int32_t>(1 << 15) - 1, sum)));
  }
}

template<size_t HEIGHT, size_t COMBINED_WIDTH, size_t WIDTH1, size_t WIDTH2>
inline void concat_and_matmul_int8(const Matrix<HEIGHT, COMBINED_WIDTH, int8_t>& mat, const Vector<WIDTH1, int8_t>& vec1, const Vector<WIDTH2, int8_t>& vec2, Vector<HEIGHT, int16_t>* out) {
  static_assert(WIDTH1 + WIDTH2 == COMBINED_WIDTH, "Matrix width must match the sum of the two vector widths");
  for (size_t i = 0; i < HEIGHT; ++i) {
    int32_t sum = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    // int8x16_t loads 16 elements at once. vdotq_s32 processes 16 MACs per instruction.
    for (size_t j = 0; j < WIDTH1; j += 64) {
      int8x16_t m0 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + j]);
      int8x16_t v0 = vld1q_s8(&vec1.data[j]);
      int8x16_t m1 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + j + 16]);
      int8x16_t v1 = vld1q_s8(&vec1.data[j + 16]);
      int8x16_t m2 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + j + 32]);
      int8x16_t v2 = vld1q_s8(&vec1.data[j + 32]);
      int8x16_t m3 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + j + 48]);
      int8x16_t v3 = vld1q_s8(&vec1.data[j + 48]);

      sum0 = vdotq_s32(sum0, m0, v0);
      sum1 = vdotq_s32(sum1, m1, v1);
      sum2 = vdotq_s32(sum2, m2, v2);
      sum3 = vdotq_s32(sum3, m3, v3);
    }
    for (size_t j = 0; j < WIDTH2; j += 64) {
      int8x16_t m0 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j]);
      int8x16_t v0 = vld1q_s8(&vec2.data[j]);
      int8x16_t m1 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 16]);
      int8x16_t v1 = vld1q_s8(&vec2.data[j + 16]);
      int8x16_t m2 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 32]);
      int8x16_t v2 = vld1q_s8(&vec2.data[j + 32]);
      int8x16_t m3 = vld1q_s8(&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 48]);
      int8x16_t v3 = vld1q_s8(&vec2.data[j + 48]);

      sum0 = vdotq_s32(sum0, m0, v0);
      sum1 = vdotq_s32(sum1, m1, v1);
      sum2 = vdotq_s32(sum2, m2, v2);
      sum3 = vdotq_s32(sum3, m3, v3);
    }
    int32x4_t sum_vec = vaddq_s32(vaddq_s32(sum0, sum1), vaddq_s32(sum2, sum3));
    sum = vaddvq_s32(sum_vec);
#else
    for (size_t j = 0; j < WIDTH1; ++j) {
      sum += static_cast<int32_t>(mat.data[i * COMBINED_WIDTH + j]) * static_cast<int32_t>(vec1.data[j]);
    }
    for (size_t j = 0; j < WIDTH2; ++j) {
      sum += static_cast<int32_t>(mat.data[i * COMBINED_WIDTH + WIDTH1 + j]) * static_cast<int32_t>(vec2.data[j]);
    }
#endif
    // The inputs were scaled by 2 instead of 0 (dropped 2 bits for quant). The weights were scaled by 6 instead of 8.
    // So the product is scaled by 8, not 16. Wait, wait:
    // Original int16_t input: S8. Shift right by 2 -> S6.
    // Original int16_t weight: S8. Shift right by 2 -> S6.
    // Product of S6 * S6 = S12.
    // Normal input is S8 * S8 = S16, then we shift right by SCALE_SHIFT (8) = S8.
    // To get S12 back down to S8, we must shift right by exactly 4.
    sum >>= 4;
    out->data[i] = static_cast<int16_t>(std::max(-(1 << 15), std::min(static_cast<int32_t>(1 << 15) - 1, sum)));
  }
}

template<size_t HEIGHT, size_t COMBINED_WIDTH, size_t WIDTH1, size_t WIDTH2>
inline void concat_and_matmul(const Matrix<HEIGHT, COMBINED_WIDTH, float>& mat, const Vector<WIDTH1, float>& vec1, const Vector<WIDTH2, float>& vec2, Vector<HEIGHT, float>* out) {
  static_assert(WIDTH1 + WIDTH2 == COMBINED_WIDTH, "Matrix width must match the sum of the two vector widths");
  for (size_t i = 0; i < HEIGHT; ++i) {
    float sum = 0;
    for (size_t j = 0; j < WIDTH1; ++j) {
      sum += mat.data[i * COMBINED_WIDTH + j] * vec1.data[j];
    }
    for (size_t j = 0; j < WIDTH2; ++j) {
      sum += mat.data[i * COMBINED_WIDTH + WIDTH1 + j] * vec2.data[j];
    }
    out->data[i] = sum;
  }
}

template<size_t HEIGHT, size_t COMBINED_WIDTH, size_t WIDTH1, size_t WIDTH2>
inline void concat_and_matmul(const Matrix<HEIGHT, COMBINED_WIDTH, int8_t>& mat, const Vector<WIDTH1, float>& vec1, const Vector<WIDTH2, float>& vec2, Vector<HEIGHT, float>* out) {
  static_assert(WIDTH1 + WIDTH2 == COMBINED_WIDTH, "Matrix width must match the sum of the two vector widths");
  for (size_t i = 0; i < HEIGHT; ++i) {
    float sum = 0;
    for (size_t j = 0; j < WIDTH1; ++j) {
      sum += static_cast<float>(mat.data[i * COMBINED_WIDTH + j]) * vec1.data[j];
    }
    for (size_t j = 0; j < WIDTH2; ++j) {
      sum += static_cast<float>(mat.data[i * COMBINED_WIDTH + WIDTH1 + j]) * vec2.data[j];
    }
    out->data[i] = sum / static_cast<float>(1 << (SCALE_SHIFT - 2));
  }
}

template<typename T>
struct Nnue {
  Vector<EMBEDDING_DIM, T> embWeights[NNUE_INPUT_DIM];

  Matrix<HIDDEN1_DIM, EMBEDDING_DIM * 2, int8_t> layer1;
  Vector<HIDDEN1_DIM, T> bias1;
  Vector<HIDDEN1_DIM, T> hidden1;

  Matrix<OUTPUT_DIM, HIDDEN1_DIM, T> layer2;
  Vector<OUTPUT_DIM, T> bias2;
  Vector<OUTPUT_DIM, T> output;

  Vector<EMBEDDING_DIM, T> clippedMover;
  Vector<EMBEDDING_DIM, T> clippedOpponent;

  Nnue() {
    layer1.setZero();
    bias1.setZero();
    layer2.setZero();
    bias2.setZero();
    output.setZero();
  }

  void increment(Vector<EMBEDDING_DIM, T> *whiteAcc, Vector<EMBEDDING_DIM, T> *blackAcc, size_t index) {
    *whiteAcc += embWeights[index];
    *blackAcc += embWeights[flip_feature_index(index)];
  }

  void decrement(Vector<EMBEDDING_DIM, T> *whiteAcc, Vector<EMBEDDING_DIM, T> *blackAcc, size_t index) {
    *whiteAcc -= embWeights[index];
    *blackAcc -= embWeights[flip_feature_index(index)];
  }

  void randn_() {
    for (size_t i = 0; i < NNUE_INPUT_DIM; ++i) {
      embWeights[i].randn_();
    }
    layer1.randn_();
    bias1.randn_();
    layer2.randn_();
    bias2.randn_();
  }

  void load(std::istream& in) {
    auto emb = std::make_unique<Matrix<NNUE_INPUT_DIM, EMBEDDING_DIM, T>>();
    emb->load_from_stream(in);
    for (size_t i = 0; i < NNUE_INPUT_DIM; ++i) {
      embWeights[i].load_from_row(*emb, i);
    }
    layer1.load_from_stream(in);
    bias1.load_from_stream(in);
    layer2.load_from_stream(in);
    bias2.load_from_stream(in);

    // Verify that the entire file has been read
    char dummy;
    if (in.read(&dummy, 1) || !in.eof()) {
      throw std::runtime_error("File not completely read");
    }
  }

  void use_debug_weights() {
    for (size_t i = 0; i < NNUE_INPUT_DIM; ++i) {
      embWeights[i].setZero();
      embWeights[i].data[i] = static_cast<int16_t>(1);
    }
    layer1.randn_();
    bias1.randn_();
    layer2.randn_();
    bias2.randn_();
  }

  T *forward(const Vector<EMBEDDING_DIM, T>& mover, const Vector<EMBEDDING_DIM, T>& opponent) {
    this->clippedMover = mover;
    this->clippedOpponent = opponent;
    
    constexpr T maxValue = std::is_same<T, float>::value ? T(1) : T(1 << SCALE_SHIFT);
    this->clippedMover.clip_(T(0), maxValue);
    this->clippedOpponent.clip_(T(0), maxValue);
    
    if constexpr (std::is_same<T, int16_t>::value) {
      Vector<EMBEDDING_DIM, int8_t> qmover;
      Vector<EMBEDDING_DIM, int8_t> qopponent;
#if defined(__ARM_NEON) || defined(__aarch64__)
      for (size_t j = 0; j < EMBEDDING_DIM; j += 16) {
        int16x8_t m0 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedMover.data[j]));
        int16x8_t m1 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedMover.data[j + 8]));
        int8x8_t mq0 = vqshrn_n_s16(m0, 2);
        int8x8_t mq1 = vqshrn_n_s16(m1, 2);
        vst1q_s8(&qmover.data[j], vcombine_s8(mq0, mq1));

        int16x8_t o0 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedOpponent.data[j]));
        int16x8_t o1 = vld1q_s16(reinterpret_cast<const int16_t*>(&clippedOpponent.data[j + 8]));
        int8x8_t oq0 = vqshrn_n_s16(o0, 2);
        int8x8_t oq1 = vqshrn_n_s16(o1, 2);
        vst1q_s8(&qopponent.data[j], vcombine_s8(oq0, oq1));
      }
#else
      for (size_t j = 0; j < EMBEDDING_DIM; ++j) {
        qmover.data[j] = std::max<int16_t>(-(1 << 7), std::min<int16_t>((1 << 7) - 1, clippedMover.data[j] >> 2));
        qopponent.data[j] = std::max<int16_t>(-(1 << 7), std::min<int16_t>((1 << 7) - 1, clippedOpponent.data[j] >> 2));
      }
#endif
      concat_and_matmul_int8<HIDDEN1_DIM, EMBEDDING_DIM * 2, EMBEDDING_DIM, EMBEDDING_DIM>(
          layer1, qmover, qopponent, reinterpret_cast<Vector<HIDDEN1_DIM, int16_t>*>(&hidden1));
    } else {
      // Float implementation is for debugging. It is not intended to be highly optimized.
      concat_and_matmul<HIDDEN1_DIM, EMBEDDING_DIM * 2, EMBEDDING_DIM, EMBEDDING_DIM>(
          layer1, clippedMover, clippedOpponent, reinterpret_cast<Vector<HIDDEN1_DIM, float>*>(&hidden1));
    }
  
    hidden1 += bias1;
    hidden1.clip_(T(0), maxValue);

    matmul(layer2, hidden1, &output);
    this->output += bias2;
    return this->output.data_ptr();
  }

  std::shared_ptr<Nnue> clone() const {
    std::shared_ptr<Nnue> copy = std::make_shared<Nnue>();
    for (size_t i = 0; i < NNUE_INPUT_DIM; ++i) {
      copy->embWeights[i] = this->embWeights[i];
    }
    copy->layer1 = this->layer1;
    copy->bias1 = this->bias1;
    copy->layer2 = this->layer2;
    copy->bias2 = this->bias2;
    return copy;
  }
};

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUE_H
