#ifndef SRC_STRING_UTILS_H
#define SRC_STRING_UTILS_H

#include <iostream>

template<class T>
std::string join(const T& A, const std::string& delimiter) {
  std::string r = "";
  for (size_t i = 0; i < A.size(); ++i) {
    r += A[i];
    if (i != A.size() - 1) {
      r += delimiter;
    }
  }
  return r;
}

std::string repeat(const std::string& s, size_t n);
std::string repr(const std::string&);
std::string rjust(const std::string& text, size_t width);
void ltrim(std::string *);
void rtrim(std::string *);
void remove_excess_whitespace(std::string *);

#endif  // SRC_STRING_UTILS_H
