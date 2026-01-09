#ifndef SRC_GAME_MOVE_H
#define SRC_GAME_MOVE_H

#include <cstdint>

#include "geometry.h"

namespace ChessEngine {

enum MoveType {
  NORMAL = 0,
  EN_PASSANT = 1,
  CASTLE = 2,
  PROMOTION = 3,
};

struct Move {
  SafeSquare from : 6;
  SafeSquare to : 6;
  unsigned promotion : 2;  // knight, bishop, rook, queen
  MoveType moveType : 2;

  std::string uci() const;

  bool operator==(const Move& a) const {
    return from == a.from && to == a.to && promotion == a.promotion && moveType == a.moveType;
  }

  static Move fromUci(const std::string& uci);
};

struct ExtMove {
  ExtMove() {}
  ExtMove(Piece piece, Move move) : piece(piece), capture(ColoredPiece::NO_COLORED_PIECE), move(move) {}
  ExtMove(Piece piece, ColoredPiece capture2, Move move) : piece(piece), capture(capture2), move(move) {}

  std::string str() const;

  std::string uci() const;

  Piece piece : 4;
  ColoredPiece capture : 4;
  Move move;  // 16 bits
  Evaluation score;  // 16 bits
};


std::ostream& operator<<(std::ostream& stream, const Move move);
std::ostream& operator<<(std::ostream& stream, const ExtMove move);

const Move kNullMove = Move{SafeSquare(0), SafeSquare(0), 0, MoveType::NORMAL};
const ExtMove kNullExtMove = ExtMove(Piece::NO_PIECE, kNullMove);

}  // namespace ChessEngine

// Hash specialization for Move to allow use in unordered containers
namespace std {
template <>
struct hash<ChessEngine::Move> {
  size_t operator()(const ChessEngine::Move& m) const {
    // Pack all fields into a single value for hashing
    return hash<uint16_t>()(
        (static_cast<uint16_t>(m.from) << 10) |
        (static_cast<uint16_t>(m.to) << 4) |
        (static_cast<uint16_t>(m.promotion) << 2) |
        static_cast<uint16_t>(m.moveType));
  }
};
}  // namespace std

#endif  // SRC_GAME_MOVE_H