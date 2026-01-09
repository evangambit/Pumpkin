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

Move Move::fromUci(const std::string& uci) {
  if (uci == "NULL") {
    return kNullMove;
  }
  if (uci.length() < 4) {
    throw std::invalid_argument("Invalid UCI move string");
  }
  SafeSquare from = string_to_safe_square(uci.substr(0, 2));
  SafeSquare to = string_to_safe_square(uci.substr(2, 2));
  MoveType moveType = MoveType::NORMAL;
  unsigned promotion = 0;
  if (uci.length() == 5 && uci[4] != '+' && uci[4] != '#') {
    if (uci[4] != 'n' && uci[4] != 'b' && uci[4] != 'r' && uci[4] != 'q') {
      throw std::invalid_argument("Invalid promotion piece in UCI");
    }
    moveType = MoveType::PROMOTION;
    char promo_char = uci[4];
    switch (promo_char) {
      case 'n': promotion = 0; break;
      case 'b': promotion = 1; break;
      case 'r': promotion = 2; break;
      case 'q': promotion = 3; break;
      default: throw std::invalid_argument("Invalid promotion piece in UCI");
    }
  }
  return Move{from, to, promotion, moveType};
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