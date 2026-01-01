#include "movegen.h"

namespace ChessEngine {

bool colorless_is_stalemate(Position *pos) {
  if (pos->turn_ == Color::WHITE) {
    return is_stalemate<Color::WHITE>(pos);
  } else {
    return is_stalemate<Color::BLACK>(pos);
  }
}

}  // namespace ChessEngine
