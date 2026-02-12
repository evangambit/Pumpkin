#include "../game/geometry.h"

namespace ChessEngine {

// Oriented so our pawns are always moving north.
struct OrientedBitboard {
  Bitboard bits;
  OrientedBitboard operator|(const OrientedBitboard& other) const {
    return OrientedBitboard{bits | other.bits};
  }
  OrientedBitboard operator&(const OrientedBitboard& other) const {
    return OrientedBitboard{bits & other.bits};
  }
  OrientedBitboard iff(bool bit) const {
    return OrientedBitboard{bit ? bits : 0};
  }
};

template<Direction DIR>
inline OrientedBitboard shift(OrientedBitboard ob) {
  return OrientedBitboard{shift<DIR>(ob.bits)};
}

template<Color US>
OrientedBitboard orient(Bitboard b) {
  if constexpr (US == Color::BLACK) {
    return OrientedBitboard{flip_vertically(b)};
  } else {
    return OrientedBitboard{b};
  }
}

inline OrientedBitboard flip_vertically(OrientedBitboard ob) {
  return OrientedBitboard{flip_vertically(ob.bits)};
}

inline SafeSquare lsb_i_promise_board_is_not_empty(OrientedBitboard ob) {
  return lsb_i_promise_board_is_not_empty(ob.bits);
}

}  // namespace ChessEngine