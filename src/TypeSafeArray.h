#ifndef TYPE_SAFE_ARRAY_H
#define TYPE_SAFE_ARRAY_H

#include <initializer_list>
#include <algorithm>
namespace ChessEngine {

template<typename T, size_t NUM_BOARDS, typename INDEX_TYPE>
struct TypeSafeArray {
  TypeSafeArray() {}
  TypeSafeArray(std::initializer_list<T> values) {
    std::copy(values.begin(), values.end(), values_);
  }
  void fill(const T& value) {
    std::fill_n(values_, NUM_BOARDS, value);
  }

  inline T& operator[](INDEX_TYPE index) {
    return values_[index];
  }

  inline const T& operator[](INDEX_TYPE index) const {
    return values_[index];
  }

 private:
  T values_[NUM_BOARDS];
};

}  // namespace ChessEngine

#endif  // TYPE_SAFE_ARRAY_H