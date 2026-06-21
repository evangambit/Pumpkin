#ifndef SRC_EVAL_NNUE_MATRIX_H
#define SRC_EVAL_NNUE_MATRIX_H

#include "NnueFeatureBitmapType.h"
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../../game/Position.h"
#include "Utils.h"

namespace NNUE {

template <size_t HEIGHT, size_t WIDTH, typename T>
struct Matrix {
  static_assert(WIDTH > 0 && HEIGHT > 0, "Matrix dimensions must be greater than zero");
  static_assert(std::is_same<T, int16_t>::value || std::is_same<T, int8_t>::value || std::is_same<T, float>::value, "Matrix type must be int16_t, int8_t, or float");
  alignas(32) T data[HEIGHT * WIDTH];

  Matrix() {
    setZero();
  }

  ~Matrix() = default;

  Matrix& operator=(const Matrix& other) = default;

  Matrix(const Matrix& other) = default;

  Matrix& operator=(Matrix&& other) = default;

  Matrix(Matrix&& other) = default;

  T operator()(size_t y, size_t x) const {
    return data[y * WIDTH + x];
  }

  T& operator()(size_t y, size_t x) {
    return data[y * WIDTH + x];
  }

  void setZero() {
    std::fill(data, data + HEIGHT * WIDTH, T(0));
  }

  void load_from_stream(std::istream& in, std::string expectedName = "", bool pad_with_zeroes_if_missing_columns = false) {
    char name[16];
    in.read(name, 16);
    if (expectedName.size() > 0) {
      while (expectedName.size() < 16) {
        expectedName += " ";
      }
      std::string actualName(name, 16);
      // Compare names.
      if (actualName != expectedName) {
        throw std::runtime_error("Unexpected matrix name: \"" + std::string(actualName) + "\" != \"" + expectedName + "\"");
      }
    }

    uint32_t degree;
    in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
    if (degree != 2) {
      throw std::runtime_error("Only 2D matrices are supported; cannot load matrix with degree " + std::to_string(degree));
    }
    uint32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    if (rows != HEIGHT || (cols != WIDTH && !pad_with_zeroes_if_missing_columns)) {
      throw std::runtime_error("Matrix size mismatch; expected " + std::to_string(HEIGHT) + "x" + std::to_string(WIDTH) + ", got " + std::to_string(rows) + "x" + std::to_string(cols));
    }
    float *buffer = new float[rows * cols];
    in.read(reinterpret_cast<char*>(buffer), sizeof(float) * rows * cols);
    for (size_t i = 0; i < HEIGHT; ++i) {
      for (size_t j = 0; j < WIDTH; ++j) {
        if (j >= cols && pad_with_zeroes_if_missing_columns) {
          data[i * WIDTH + j] = T(0);
          continue;
        }
        if (std::is_same<T, float>::value) {
          data[i * WIDTH + j] = buffer[i * cols + j];
        } else if (std::is_same<T, int16_t>::value) {
          data[i * WIDTH + j] = static_cast<T>(buffer[i * cols + j] * (1 << SCALE_SHIFT));
        } else if (std::is_same<T, int8_t>::value) {
          // Quantize weights down to fit perfectly inside an int8_t
          // SCALE_SHIFT is 8, meaning we scale floats by 256. We need to scale by 64 instead (SCALE_SHIFT - 2)
          data[i * WIDTH + j] = static_cast<T>(buffer[i * cols + j] * (1 << (SCALE_SHIFT - 2)));
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
  alignas(32) T data[DIM];

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

  void load_from_stream(std::istream& in, std::string expectedName = "") {
    char name[16];
    in.read(name, 16);
    if (expectedName.size() > 0) {
      while (expectedName.size() < 16) {
        expectedName += " ";
      }
      std::string actualName(name, 16);
      // Compare names.
      if (actualName != expectedName) {
        throw std::runtime_error("Unexpected matrix name: \"" + std::string(actualName) + "\" != \"" + expectedName + "\"");
      }
    }

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
#elif defined(__AVX2__)
    if constexpr (std::is_same<T, int16_t>::value) {
      for (size_t i = 0; i < DIM; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i*)&data[i]);
        __m256i b = _mm256_load_si256((const __m256i*)&other.data[i]);
        _mm256_store_si256((__m256i*)&data[i], _mm256_add_epi16(a, b));
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
#elif defined(__AVX2__)
    if constexpr (std::is_same<T, int16_t>::value) {
      for (size_t i = 0; i < DIM; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i*)&data[i]);
        __m256i b = _mm256_load_si256((const __m256i*)&other.data[i]);
        _mm256_store_si256((__m256i*)&data[i], _mm256_sub_epi16(a, b));
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
#elif defined(__AVX2__)
    if constexpr (std::is_same<T, int16_t>::value) {
      __m256i vMin = _mm256_set1_epi16(minVal);
      __m256i vMax = _mm256_set1_epi16(maxVal);
      for (size_t i = 0; i < DIM; i += 16) {
        __m256i v = _mm256_load_si256((const __m256i*)&data[i]);
        v = _mm256_max_epi16(vMin, _mm256_min_epi16(v, vMax));
        _mm256_store_si256((__m256i*)&data[i], v);
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
    int32x4_t sum_vec = vdupq_n_s32(0);
    size_t j = 0;
    for (; j + 31 < WIDTH; j += 32) {
      int16x8_t m0 = vld1q_s16(&mat.data[i * WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec.data[j]);
      int16x8_t m1 = vld1q_s16(&mat.data[i * WIDTH + j + 8]);
      int16x8_t v1 = vld1q_s16(&vec.data[j + 8]);
      int16x8_t m2 = vld1q_s16(&mat.data[i * WIDTH + j + 16]);
      int16x8_t v2 = vld1q_s16(&vec.data[j + 16]);
      int16x8_t m3 = vld1q_s16(&mat.data[i * WIDTH + j + 24]);
      int16x8_t v3 = vld1q_s16(&vec.data[j + 24]);

      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m0), vget_low_s16(v0));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m0), vget_high_s16(v0));
      
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m1), vget_low_s16(v1));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m1), vget_high_s16(v1));

      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m2), vget_low_s16(v2));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m2), vget_high_s16(v2));

      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m3), vget_low_s16(v3));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m3), vget_high_s16(v3));
    }
    for (; j + 15 < WIDTH; j += 16) {
      int16x8_t m0 = vld1q_s16(&mat.data[i * WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec.data[j]);
      int16x8_t m1 = vld1q_s16(&mat.data[i * WIDTH + j + 8]);
      int16x8_t v1 = vld1q_s16(&vec.data[j + 8]);
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m0), vget_low_s16(v0));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m0), vget_high_s16(v0));
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m1), vget_low_s16(v1));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m1), vget_high_s16(v1));
    }
    for (; j + 7 < WIDTH; j += 8) {
      int16x8_t m0 = vld1q_s16(&mat.data[i * WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec.data[j]);
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m0), vget_low_s16(v0));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m0), vget_high_s16(v0));
    }
    sum = vaddvq_s32(sum_vec);
    for (; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#elif defined(__AVX2__)
    __m256i sum0 = _mm256_setzero_si256();
    size_t j = 0;
    for (; j + 31 < WIDTH; j += 32) {
      __m256i m0 = _mm256_load_si256((const __m256i*)&mat.data[i * WIDTH + j]);
      __m256i v0 = _mm256_load_si256((const __m256i*)&vec.data[j]);
      __m256i m1 = _mm256_load_si256((const __m256i*)&mat.data[i * WIDTH + j + 16]);
      __m256i v1 = _mm256_load_si256((const __m256i*)&vec.data[j + 16]);

      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(m0, v0));
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(m1, v1));
    }
    for (; j + 15 < WIDTH; j += 16) {
      __m256i m0 = _mm256_load_si256((const __m256i*)&mat.data[i * WIDTH + j]);
      __m256i v0 = _mm256_load_si256((const __m256i*)&vec.data[j]);
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(m0, v0));
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum0), _mm256_extracti128_si256(sum0, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_cvtsi128_si32(sum128);
    for (; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#else
    for (size_t j = 0; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#endif
    sum >>= SCALE_SHIFT;
    out->data[i] = static_cast<int16_t>(std::max(-(1 << 15), std::min(static_cast<int32_t>(1 << 15) - 1, sum)));
  }
}

template<size_t HEIGHT, size_t WIDTH>
inline void matmul(Matrix<HEIGHT, WIDTH, int8_t>& mat, const Vector<WIDTH, int16_t>& vec, Vector<HEIGHT, int16_t>* out) {
  for (size_t i = 0; i < HEIGHT; ++i) {
    int32_t sum = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    int32x4_t sum_vec = vdupq_n_s32(0);
    size_t j = 0;
    for (; j + 15 < WIDTH; j += 16) {
      int8x8_t m0 = vld1_s8(&mat.data[i * WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec.data[j]);
      int8x8_t m1 = vld1_s8(&mat.data[i * WIDTH + j + 8]);
      int16x8_t v1 = vld1q_s16(&vec.data[j + 8]);

      // Multiply int8 * int16 to get int32
      int16x8_t m0_ext = vmovl_s8(m0);
      int16x8_t m1_ext = vmovl_s8(m1);
      
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m0_ext), vget_low_s16(v0));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m0_ext), vget_high_s16(v0));
      
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m1_ext), vget_low_s16(v1));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m1_ext), vget_high_s16(v1));
    }
    for (; j + 7 < WIDTH; j += 8) {
      int8x8_t m0 = vld1_s8(&mat.data[i * WIDTH + j]);
      int16x8_t v0 = vld1q_s16(&vec.data[j]);
      
      int16x8_t m0_ext = vmovl_s8(m0);
      sum_vec = vmlal_s16(sum_vec, vget_low_s16(m0_ext), vget_low_s16(v0));
      sum_vec = vmlal_s16(sum_vec, vget_high_s16(m0_ext), vget_high_s16(v0));
    }
    sum = vaddvq_s32(sum_vec);
    for (; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#elif defined(__AVX2__)
    __m256i sum0 = _mm256_setzero_si256();
    size_t j = 0;
    for (; j + 31 < WIDTH; j += 32) {
      // Load 32 int8_t values from matrix
      __m256i m0 = _mm256_load_si256((const __m256i*)&mat.data[i * WIDTH + j]);
      // Load 16 int16_t values (first half)
      __m256i v0 = _mm256_load_si256((const __m256i*)&vec.data[j]);
      // Load 16 int16_t values (second half)
      __m256i v1 = _mm256_load_si256((const __m256i*)&vec.data[j + 16]);
      
      // Sign-extend int8 to int16
      __m256i m0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(m0));
      __m256i m0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(m0, 1));
      
      // Multiply int16 * int16 to get int32
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(m0_lo, v0));
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(m0_hi, v1));
    }
    for (; j + 15 < WIDTH; j += 16) {
      __m128i m0 = _mm_load_si128((const __m128i*)&mat.data[i * WIDTH + j]);
      __m256i v0 = _mm256_load_si256((const __m256i*)&vec.data[j]);
      
      __m256i m0_ext = _mm256_cvtepi8_epi16(m0);
      sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(m0_ext, v0));
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum0), _mm256_extracti128_si256(sum0, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_cvtsi128_si32(sum128);
    for (; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#else
    for (size_t j = 0; j < WIDTH; ++j) {
      sum += static_cast<int32_t>(mat.data[i * WIDTH + j]) * static_cast<int32_t>(vec.data[j]);
    }
#endif
    sum >>= SCALE_SHIFT;
    out->data[i] = static_cast<int16_t>(std::max(-(1 << 15), std::min(static_cast<int32_t>(1 << 15) - 1, sum)));
  }
}

template<size_t HEIGHT, size_t WIDTH>
inline void matmul(Matrix<HEIGHT, WIDTH, int8_t>& mat, const Vector<WIDTH, float>& vec, Vector<HEIGHT, float>* out) {
  for (size_t i = 0; i < HEIGHT; ++i) {
    float sum = 0;
    for (size_t j = 0; j < WIDTH; ++j) {
      sum += static_cast<float>(mat.data[i * WIDTH + j]) * vec.data[j];
    }
    out->data[i] = sum;
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
#elif defined(__AVX2__)
    __m256i sum0_avx = _mm256_setzero_si256();
    __m256i sum1_avx = _mm256_setzero_si256();
    for (size_t j = 0; j < WIDTH1; j += 32) {
      __m256i m0 = _mm256_load_si256((const __m256i*)&mat.data[i * COMBINED_WIDTH + j]);
      __m256i v0 = _mm256_load_si256((const __m256i*)&vec1.data[j]);
      __m256i m1 = _mm256_load_si256((const __m256i*)&mat.data[i * COMBINED_WIDTH + j + 16]);
      __m256i v1 = _mm256_load_si256((const __m256i*)&vec1.data[j + 16]);

      sum0_avx = _mm256_add_epi32(sum0_avx, _mm256_madd_epi16(m0, v0));
      sum1_avx = _mm256_add_epi32(sum1_avx, _mm256_madd_epi16(m1, v1));
    }
    for (size_t j = 0; j < WIDTH2; j += 32) {
      __m256i m0 = _mm256_load_si256((const __m256i*)&mat.data[i * COMBINED_WIDTH + WIDTH1 + j]);
      __m256i v0 = _mm256_load_si256((const __m256i*)&vec2.data[j]);
      __m256i m1 = _mm256_load_si256((const __m256i*)&mat.data[i * COMBINED_WIDTH + WIDTH1 + j + 16]);
      __m256i v1 = _mm256_load_si256((const __m256i*)&vec2.data[j + 16]);

      sum0_avx = _mm256_add_epi32(sum0_avx, _mm256_madd_epi16(m0, v0));
      sum1_avx = _mm256_add_epi32(sum1_avx, _mm256_madd_epi16(m1, v1));
    }
    __m256i sum_vec_avx = _mm256_add_epi32(sum0_avx, sum1_avx);
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum_vec_avx), _mm256_extracti128_si256(sum_vec_avx, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_cvtsi128_si32(sum128);
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
#elif defined(__AVX2__)
    __m256i sum0_avx = _mm256_setzero_si256();
    __m256i sum1_avx = _mm256_setzero_si256();
    for (size_t j = 0; j < WIDTH1; j += 32) {
      __m256i m_256 = _mm256_load_si256((const __m256i*)&mat.data[i * COMBINED_WIDTH + j]);
      __m256i m0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(m_256));
      __m256i m1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(m_256, 1));
      
      __m256i v_256 = _mm256_load_si256((const __m256i*)&vec1.data[j]);
      __m256i v0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v_256));
      __m256i v1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v_256, 1));

      sum0_avx = _mm256_add_epi32(sum0_avx, _mm256_madd_epi16(m0, v0));
      sum1_avx = _mm256_add_epi32(sum1_avx, _mm256_madd_epi16(m1, v1));
    }
    for (size_t j = 0; j < WIDTH2; j += 32) {
      __m256i m_256 = _mm256_load_si256((const __m256i*)&mat.data[i * COMBINED_WIDTH + WIDTH1 + j]);
      __m256i m0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(m_256));
      __m256i m1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(m_256, 1));
      
      __m256i v_256 = _mm256_load_si256((const __m256i*)&vec2.data[j]);
      __m256i v0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v_256));
      __m256i v1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v_256, 1));

      sum0_avx = _mm256_add_epi32(sum0_avx, _mm256_madd_epi16(m0, v0));
      sum1_avx = _mm256_add_epi32(sum1_avx, _mm256_madd_epi16(m1, v1));
    }
    __m256i sum_vec_avx = _mm256_add_epi32(sum0_avx, sum1_avx);
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum_vec_avx), _mm256_extracti128_si256(sum_vec_avx, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_cvtsi128_si32(sum128);
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

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_MATRIX_H