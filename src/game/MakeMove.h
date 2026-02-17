#ifndef MAKE_MOVE_H
#define MAKE_MOVE_H

#include "Position.h"

namespace ChessEngine {

template<Color TURN>
void make_nullmove(Position *pos) {
  pos->states_.push_back(pos->currentState_);

  pos->history_.push_back(ExtMove(Piece::NO_PIECE, ColoredPiece::NO_COLORED_PIECE, kNullMove));

  const UnsafeSquare oldEpSquare = pos->currentState_.epSquare;
  pos->currentState_.epSquare = UnsafeSquare::UNO_SQUARE;

  if (TURN == Color::BLACK) {
    pos->wholeMoveCounter_ += 1;
  }
  ++pos->currentState_.halfMoveCounter;
  pos->turn_ = opposite_color<TURN>();
  pos->currentState_.hash ^= kZorbristTurn;
  pos->currentState_.hash ^= kZorbristEnpassant[oldEpSquare % 8] * (oldEpSquare != UnsafeSquare::UNO_SQUARE);
}

template<Color MOVER_TURN>
void undo_nullmove(Position *pos) {
  pos->currentState_ = pos->states_.back();
  pos->states_.pop_back();

  assert(pos->history_.back().move == kNullMove);
  pos->history_.pop_back();

  if (MOVER_TURN == Color::BLACK) {
    pos->wholeMoveCounter_ -= 1;
  }

  pos->turn_ = MOVER_TURN;
}

template<Color TURN>
void make_move(Position *pos, Move move) {
  foo<TURN>(pos);
  assert_valid_square(move.to);
  assert(cp2color(pos->tiles_[move.to]) != TURN);

  const ColoredPiece movingPiece = pos->tiles_[move.from];
  const ColoredPiece promoPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<TURN>(Piece(move.promotion + 2)) : movingPiece;
  constexpr Color opposingColor = opposite_color<TURN>();
  const ColoredPiece capturedPiece = pos->tiles_[move.to];

  const Location f = square2location(move.from);
  const Location t = square2location(move.to);

  pos->states_.push_back(pos->currentState_);
  pos->history_.push_back(ExtMove(cp2p(movingPiece), capturedPiece, move));

  // Remove castling rights if a rook moves or is captured.
  CastlingRights newCastlingRights = pos->currentState_.castlingRights;
  newCastlingRights &= ~four_corners_to_byte(f);
  newCastlingRights &= ~four_corners_to_byte(t);

  // TODO: Set epSquare to NO_SQUARE if there is now way your opponent can play en passant next move.
  //       This will make it easier to count 3-fold draw.
  const UnsafeSquare oldEpSquare = pos->currentState_.epSquare;
  if (TURN == Color::WHITE) {
    bool cond = (movingPiece == coloredPiece<TURN, Piece::PAWN>() && move.from - move.to == 16);
    pos->currentState_.epSquare = UnsafeSquare(cond * (move.to + 8) + (1 - cond) * UnsafeSquare::UNO_SQUARE);
  } else {
    bool cond = (movingPiece == coloredPiece<TURN, Piece::PAWN>() && move.to - move.from == 16);
    pos->currentState_.epSquare = UnsafeSquare(cond * (move.to - 8) + (1 - cond) * UnsafeSquare::UNO_SQUARE);
  }
  pos->currentState_.hash ^= (
    (kZorbristEnpassant[pos->currentState_.epSquare % 8] * (pos->currentState_.epSquare != UnsafeSquare::UNO_SQUARE))
    ^
    (kZorbristEnpassant[oldEpSquare % 8] * (oldEpSquare != UnsafeSquare::UNO_SQUARE))
  );

  // Remove castling rights if a king moves from its starting square.
  // Only check the current player's king starting square, not both.
  constexpr Bitboard ourKingStart = (TURN == Color::WHITE) ? bb(SafeSquare::SE1) : bb(SafeSquare::SE8);
  newCastlingRights &= ~(((
    (f & ourKingStart) > 0)
    |
    (((f & ourKingStart) > 0) << 1)
  ) << ((2 - TURN) * 2));
  pos->currentState_.hash ^= kZorbristCastling[pos->currentState_.castlingRights] ^ kZorbristCastling[newCastlingRights];
  pos->currentState_.castlingRights = newCastlingRights;


  // Move the piece.
  pos->pieceBitboards_[movingPiece] &= ~f;
  pos->pieceBitboards_[promoPiece] |= t;
  pos->colorBitboards_[TURN] &= ~f;
  pos->colorBitboards_[TURN] |= t;
  pos->tiles_[move.to] = promoPiece;
  pos->tiles_[move.from] = ColoredPiece::NO_COLORED_PIECE;
  pos->currentState_.hash ^= kZorbristNumbers[movingPiece][move.from];
  pos->currentState_.hash ^= kZorbristNumbers[promoPiece][move.to];

  // Remove captured piece.
  pos->pieceBitboards_[capturedPiece] &= ~t;
  pos->colorBitboards_[opposingColor] &= ~t;
  const bool hasCapturedPiece = (capturedPiece != ColoredPiece::NO_COLORED_PIECE);
  pos->currentState_.hash ^= kZorbristNumbers[capturedPiece][move.to] * hasCapturedPiece;

  pos->increment_piece_map(promoPiece, move.to);
  pos->decrement_piece_map(capturedPiece, move.to);
  pos->decrement_piece_map(movingPiece, move.from);

  if (move.moveType == MoveType::CASTLE) {
    // TODO: get rid of if statement
    if (TURN == Color::BLACK) {
      assert(move.from == 4);
      assert(move.to == 2 || move.to == 6);
    } else {
      assert(move.from == 60);
      assert(move.to == 62 || move.to == 58);
    }
    SafeSquare rookDestination = SafeSquare((uint16_t(move.from) + uint16_t(move.to)) / 2);
    SafeSquare rookOrigin = SafeSquare(((uint16_t(move.to) % 8) * 7 - 14) / 4 + (TURN == Color::WHITE ? 56 : 0));

    Bitboard rookDestinationBB = bb(rookDestination);
    Bitboard rookOriginBB = bb(rookOrigin);

    constexpr ColoredPiece myRookPiece = coloredPiece<TURN, Piece::ROOK>();
    pos->pieceBitboards_[myRookPiece] &= ~rookOriginBB;
    pos->pieceBitboards_[myRookPiece] |= rookDestinationBB;
    pos->colorBitboards_[TURN] &= ~rookOriginBB;
    pos->colorBitboards_[TURN] |= rookDestinationBB;
    pos->tiles_[rookOrigin] = ColoredPiece::NO_COLORED_PIECE;
    pos->tiles_[rookDestination] = myRookPiece;
    pos->currentState_.hash ^= kZorbristNumbers[myRookPiece][rookOrigin] * hasCapturedPiece;
    pos->currentState_.hash ^= kZorbristNumbers[myRookPiece][rookDestination] * hasCapturedPiece;

    pos->increment_piece_map(myRookPiece, rookDestination);
    pos->decrement_piece_map(myRookPiece, rookOrigin);
  }

  if (UnsafeSquare(move.to) == oldEpSquare && movingPiece == coloredPiece<TURN, Piece::PAWN>()) {
    // TODO: get rid of if statement
    if (TURN == Color::BLACK) {
      assert(move.from / 8 == 4);
      assert(move.to / 8 == 5);
    } else {
      assert(move.from / 8 == 3);
      assert(move.to / 8 == 2);
    }
    SafeSquare enpassantSq = SafeSquare(TURN == Color::WHITE ? move.to + 8 : move.to - 8);
    Bitboard enpassantLocBB = bb(enpassantSq);

    constexpr ColoredPiece opposingPawn = coloredPiece<opposingColor, Piece::PAWN>();

    assert(pos->tiles_[enpassantSq] == opposingPawn);

    pos->pieceBitboards_[opposingPawn] &= ~enpassantLocBB;
    pos->colorBitboards_[opposingColor] &= ~enpassantLocBB;
    pos->tiles_[enpassantSq] = ColoredPiece::NO_COLORED_PIECE;
    pos->currentState_.hash ^= kZorbristNumbers[opposingPawn][enpassantSq];
    pos->decrement_piece_map(opposingPawn, enpassantSq);
  }

  if (TURN == Color::BLACK) {
    pos->wholeMoveCounter_ += 1;
  }
  ++pos->currentState_.halfMoveCounter;
  pos->currentState_.halfMoveCounter *= (movingPiece != coloredPiece<TURN, Piece::PAWN>() && capturedPiece == ColoredPiece::NO_COLORED_PIECE);
  pos->turn_ = opposingColor;
  pos->currentState_.hash ^= kZorbristTurn;

  bar<TURN>(pos);

}

void ez_make_move(Position *position, Move move);

void ez_undo(Position *position);

}  // namespace ChessEngine

#endif  // MAKE_MOVE_H