#ifndef COLOREDEVALUATION_H
#define COLOREDEVALUATION_H

namespace ChessEngine {

/**
  * A wrapper around a raw evaluation score which is always from the perspective of a specific color.
  *
  * This is useful to avoid accidentally mixing up scores from different perspectives.
  */
template<Color TURN>
struct ColoredEvaluation {
  int16_t value;
  ColoredEvaluation() {}
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
  bool is_mating() const {
    return value <= kLongestForcedMate || value >= -kLongestForcedMate;
  }
  ColoredEvaluation<TURN> operator+(int32_t delta) const {
    delta *= !is_mating();  // Centipawn deltas are meaningless if we're mating or being mated.
    return ColoredEvaluation<TURN>(std::clamp<int32_t>(int32_t(value) + delta, kMinEval, kMaxEval));
  }
  ColoredEvaluation<TURN> operator-(int32_t delta) const {
    delta *= !is_mating();  // // Centipawn deltas are meaningless if we're mating or being mated.
    return ColoredEvaluation<TURN>(std::clamp<int32_t>(int32_t(value) - delta, kMinEval, kMaxEval));
  }
  ColoredEvaluation<TURN>& clamp_(const ColoredEvaluation<TURN>& alpha, const ColoredEvaluation<TURN>& beta) {
    value = std::max(alpha.value, std::min(value, beta.value));
    return *this;
  }
  friend std::ostream& operator<<(std::ostream& os, const ColoredEvaluation<TURN>& eval) {
    os << (TURN == Color::WHITE ? eval.value : -eval.value);
    return os;
  }
};

}  // namespace ChessEngine

#endif  // COLOREDEVALUATION_H
