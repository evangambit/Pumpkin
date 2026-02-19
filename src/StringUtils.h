#ifndef SRC_STRING_UTILS_H
#define SRC_STRING_UTILS_H

#include <iostream>
#include <vector>

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

inline std::vector<std::string> split(const std::string& s, char delimiter) {
  std::vector<std::string> r;
  std::string current = "";
  for (char c : s) {
    if (c == delimiter) {
      r.push_back(current);
      current = "";
    } else {
      current += c;
    }
  }
  r.push_back(current);
  return r;
}

std::string repeat(const std::string& s, size_t n);
std::string repr(const std::string&);
std::string rjust(const std::string& text, size_t width);
void ltrim(std::string *);
void rtrim(std::string *);
void remove_excess_whitespace(std::string *);

inline std::string merge_tables(const std::vector<std::string>& tables) {
 std::vector<std::string> lines;
  for (const std::string& table : tables) {
    std::vector<std::string> rows = split(table, '\n');
    size_t width = 0;
    for (const std::string& row : rows) {
      width = std::max(width, row.size());
    }
    for (size_t i = 0; i < rows.size(); ++i) {
      if (lines.size() <= i) {
        lines.push_back("");
      }
      lines[i] += rjust(rows[i], width) + "  ";
    }
  }
  return join(lines, "\n");
}

#endif  // SRC_STRING_UTILS_H
