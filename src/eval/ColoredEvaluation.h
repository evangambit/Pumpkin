#ifndef COLOREDEVALUATION_H
#define COLOREDEVALUATION_H

namespace ChessEngine {

template<Color TURN>
struct ColoredEvaluation {
  int16_t value;
  explicit ColoredEvaluation(int16_t v) : value(v) {}
  ColoredEvaluation<opposite_color<TURN>()> operator-() const {
    return ColoredEvaluation<opposite_color<TURN>()>(-value);
  }
  bool operator>=(const ColoredEvaluation<TURN>& other) const {
    return value >= other.value;
  }
  bool operator<=(const ColoredEvaluation<TURN>& other) const {
    return value <= other.value;
  }
  bool operator>(const ColoredEvaluation<TURN>& other) const {
    return value > other.value;
  }
  bool operator<(const ColoredEvaluation<TURN>& other) const {
    return value < other.value;
  }
  bool operator==(const ColoredEvaluation<TURN>& other) const {
    return value == other.value;
  }
  bool operator!=(const ColoredEvaluation<TURN>& other) const {
    return value != other.value;
  }
  ColoredEvaluation<TURN> operator+(Evaluation other) const {
    return ColoredEvaluation<TURN>(value + other);
  }
  ColoredEvaluation<TURN>& operator+=(const ColoredEvaluation<TURN>& other) {
    value += other.value;
    return *this;
  }
  friend std::ostream& operator<<(std::ostream& os, const ColoredEvaluation<TURN>& eval) {
    os << (TURN == Color::WHITE ? eval.value : -eval.value);
    return os;
  }
};

}  // namespace ChessEngine

#endif  // COLOREDEVALUATION_H
