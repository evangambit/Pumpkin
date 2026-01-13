#include "movegen.h"

namespace ChessEngine {

bool colorless_is_stalemate(Position *pos) {
  if (pos->turn_ == Color::WHITE) {
    return is_stalemate<Color::WHITE>(pos);
  } else {
    return is_stalemate<Color::BLACK>(pos);
  }
}

Move uci_to_move(const Position& pos, const std::string& uci) {
    ExtMove moves[kMaxNumMoves];
    ExtMove *end;
    Position tempPos = pos;
    if (pos.turn_ == Color::WHITE) {
      end = compute_legal_moves<Color::WHITE>(&tempPos, moves);
    } else {
      end = compute_legal_moves<Color::BLACK>(&tempPos, moves);
    }
    for (ExtMove* m = moves; m != end; ++m) {
      Move move = m->move;
      if (move.uci() == uci) {
        return move;
      }
    }
    return kNullMove;
  }

}  // namespace ChessEngine
