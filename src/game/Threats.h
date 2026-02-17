#ifndef THREATS_H
#define THREATS_H

#include "Utils.h"
#include "Geometry.h"

namespace ChessEngine {

template<typename T, size_t NUM_BOARDS, typename INDEX_TYPE>
struct TypeSafeArray {
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

inline Bitboard at_least_two(Bitboard a, Bitboard b) {
  return a & b;
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c) {
  return (a & b) | (a & c) | (b & c);
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d) {
  // return (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d);
  // return (a & (b | c | d)) | (b & (c | d)) | (c & d);  // 8 ops
  // return ((a ^ b) & (c ^ d)) | (a & b) | (c & d);  // 7 ops
  return ((a | b) & (c | d)) | (a & b) | (c & d);  // 7 ops
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d, Bitboard e) {
  // return (a & b) | (a & c) | (a & d) | (a & e) | (b & c) | (b & d) | (b & e) | (c & d) | (c & e) | (d & e);
  // return (a & (b | c | d | e)) | (b & (c | d | e)) | (c & (d | e)) | (d & e);  // 13 ops
  return ((a | b) & (c | d | e)) | (a & b) | at_least_two(c, d, e);  // 12 ops
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d, Bitboard e, Bitboard f) {
  return ((a | b | c) & (d | e | f)) | at_least_two(a, b, c) | at_least_two(d, e, f);  // 21 ops
}

struct Threats {
  Bitboard whitePawnTargets;
  Bitboard whiteKnightTargets;
  Bitboard whiteBishopTargets;
  Bitboard whiteRookTargets;
  Bitboard whiteQueenTargets;
  Bitboard whiteKingTargets;

  Bitboard blackPawnTargets;
  Bitboard blackKnightTargets;
  Bitboard blackBishopTargets;
  Bitboard blackRookTargets;
  Bitboard blackQueenTargets;
  Bitboard blackKingTargets;

  Bitboard whiteTargets;
  Bitboard whiteDoubleTargets;
  Bitboard blackTargets;
  Bitboard blackDoubleTargets;

  // TODO: use these.
  Bitboard badForWhite[7];
  Bitboard badForBlack[7];

  template<Color US>
  Bitboard badForOur(Piece piece) const {
    if constexpr (US == Color::WHITE) {
      return badForWhite[piece];
    } else {
      return badForBlack[piece];
    }
  }

  template<ColoredPiece cp>
  Bitboard badFor() const {
    constexpr Color color = cp2color(cp);
    constexpr Piece piece = cp2p(cp);
    if (color == Color::WHITE) {
      return badForWhite[piece];
    } else {
      return badForBlack[piece];
    }
  }
};

}  // namespace ChessEngine

#endif  // THREATS_H