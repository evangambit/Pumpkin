#include "Move.h"

namespace ChessEngine {

std::string Move::uci() const {
  if (*this == kNullMove) {
    return "NULL";
  }
  std::string r = "";
  r += square_to_string(from);
  r += square_to_string(to);
  if (moveType == MoveType::PROMOTION) {
    r += piece_to_char(Piece(promotion + 2));
  }
  return r;
}

std::string ExtMove::str() const {
  if (this->move == kNullMove) {
    return "NULL";
  }
  std::string r = "";
  r += char(piece_to_char(this->piece) + 'A' - 'a');
  r += square_to_string(this->move.from);
  if (this->capture != ColoredPiece::NO_COLORED_PIECE) {
    r += "x";
  }
  r += square_to_string(this->move.to);
  if (this->move.moveType == MoveType::PROMOTION) {
    r += piece_to_char(Piece(this->move.promotion + 2));
  }
  return r;
}

std::ostream& operator<<(std::ostream& stream, const Move move) {
  stream << square_to_string(move.from) << square_to_string(move.to);
  if (move.moveType == MoveType::PROMOTION) {
    stream << piece_to_char(Piece(move.promotion + 2));
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ExtMove move) {
  return stream << move.str();
}

}  // namespace ChessEngine