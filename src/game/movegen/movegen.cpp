#include "movegen.h"

namespace ChessEngine {

Bitboard kUnencumberedMoves[Piece::NUM_PIECES][kNumSquares];

bool is_move_feasible(Piece piece, Move move) {
  return kUnencumberedMoves[piece][move.from] & bb(move.to);
}

void initialize_movegen() {
  initialize_sliding();
  for (SafeSquare sq = SafeSquare(0); sq < kNumSquares; sq = SafeSquare(sq + 1)) {
    Rank rank = square2rank(sq);
    const Bitboard b = bb(sq);
    if (rank == RANK_1 || rank == RANK_8) {
      kUnencumberedMoves[Piece::PAWN][sq] = kEmptyBitboard;
    } else {
      kUnencumberedMoves[Piece::PAWN][sq] = shift<Direction::NORTH>(b);
      kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::NORTH_EAST>(b);
      kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::NORTH_WEST>(b);
      kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::SOUTH>(b);
      kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::SOUTH_EAST>(b);
      kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::SOUTH_WEST>(b);

      if (rank == RANK_2) {
        kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::NORTHx2>(b);
      }
      if (rank == RANK_7) {
        kUnencumberedMoves[Piece::PAWN][sq] |= shift<Direction::SOUTHx2>(b);
      }
    }

    kUnencumberedMoves[Piece::KNIGHT][sq] = kKnightMoves[to_unsafe_square(sq)];
    kUnencumberedMoves[Piece::BISHOP][sq] = compute_one_bishops_targets(sq, kEmptyBitboard);
    kUnencumberedMoves[Piece::ROOK][sq] = compute_single_rook_moves(sq, kEmptyBitboard);
    kUnencumberedMoves[Piece::QUEEN][sq] = kUnencumberedMoves[Piece::BISHOP][sq] | kUnencumberedMoves[Piece::ROOK][sq];
    kUnencumberedMoves[Piece::KING][sq] = kKingMoves[sq];
    if (sq == SafeSquare::SE1 || sq == SafeSquare::SE8) {
      kUnencumberedMoves[Piece::KING][sq] |= shift<Direction::EASTx2>(b);
      kUnencumberedMoves[Piece::KING][sq] |= shift<Direction::WESTx2>(b);
    }
  }
}

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
