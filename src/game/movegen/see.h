#ifndef MOVEGEN_SEE_H
#define MOVEGEN_SEE_H

#include "../Position.h"
#include "../Utils.h"
#include "sliding.h"
#include "rooks.h"
#include "bishops.h"
#include "knights.h"
#include "kings.h"

namespace ChessEngine {

// SEE piece values (to evaluate trades).
constexpr int kSeePieceValues[Piece::NUM_PIECES] = {
  0,    // NO_PIECE
  100,  // PAWN
  320,  // KNIGHT
  330,  // BISHOP
  500,  // ROOK
  900,  // QUEEN
  20000 // KING
};

// Returns the least valuable attacker of `sq` belonging to color `color`.
// Updates `attackerPiece` and `attackerSquare` if an attacker is found.
// `occupied` provides the masking bitboard to handle X-rays seamlessly.
inline bool get_least_valuable_attacker(const Position& pos, const SafeSquare sq, const Color color, const Bitboard occupied, Piece& attackerPiece, SafeSquare& attackerSquare) {
  const Bitboard colorBB = pos.colorBitboards_[color];
  
  // 1. Pawns
  const ColoredPiece pawnPiece = (color == Color::WHITE) ? ColoredPiece::WHITE_PAWN : ColoredPiece::BLACK_PAWN;
  const Bitboard pawns = pos.pieceBitboards_[pawnPiece] & colorBB & occupied;
  if (pawns) {
    Bitboard pawnAttacksRes = 0;
    if (color == Color::WHITE) {
      pawnAttacksRes = shift<Direction::SOUTH_WEST>(bb(sq)) | shift<Direction::SOUTH_EAST>(bb(sq));
    } else {
      pawnAttacksRes = shift<Direction::NORTH_EAST>(bb(sq)) | shift<Direction::NORTH_WEST>(bb(sq));
    }
    if (pawnAttacksRes & pawns) {
      attackerSquare = lsb_i_promise_board_is_not_empty(pawnAttacksRes & pawns);
      attackerPiece = Piece::PAWN;
      return true;
    }
  }

  // 2. Knights
  const ColoredPiece knightPiece = (color == Color::WHITE) ? ColoredPiece::WHITE_KNIGHT : ColoredPiece::BLACK_KNIGHT;
  const Bitboard knights = kKnightMoves[sq] & pos.pieceBitboards_[knightPiece] & colorBB & occupied;
  if (knights) {
    attackerSquare = lsb_i_promise_board_is_not_empty(knights);
    attackerPiece = Piece::KNIGHT;
    return true;
  }

  // 3. Bishops
  const ColoredPiece bishopPiece = (color == Color::WHITE) ? ColoredPiece::WHITE_BISHOP : ColoredPiece::BLACK_BISHOP;
  Bitboard bishopAttacks = compute_one_bishops_targets(sq, occupied) & pos.pieceBitboards_[bishopPiece] & colorBB & occupied;
  if (bishopAttacks) {
    attackerSquare = lsb_i_promise_board_is_not_empty(bishopAttacks);
    attackerPiece = Piece::BISHOP;
    return true;
  }

  // 4. Rooks
  const ColoredPiece rookPiece = (color == Color::WHITE) ? ColoredPiece::WHITE_ROOK : ColoredPiece::BLACK_ROOK;
  Bitboard rookAttacks = compute_single_rook_moves(sq, occupied) & pos.pieceBitboards_[rookPiece] & colorBB & occupied;
  if (rookAttacks) {
    attackerSquare = lsb_i_promise_board_is_not_empty(rookAttacks);
    attackerPiece = Piece::ROOK;
    return true;
  }

  // 5. Queens
  const ColoredPiece queenPiece = (color == Color::WHITE) ? ColoredPiece::WHITE_QUEEN : ColoredPiece::BLACK_QUEEN;
  Bitboard queenAttacks = (compute_one_bishops_targets(sq, occupied) | compute_single_rook_moves(sq, occupied)) & pos.pieceBitboards_[queenPiece] & colorBB & occupied;
  if (queenAttacks) {
    attackerSquare = lsb_i_promise_board_is_not_empty(queenAttacks);
    attackerPiece = Piece::QUEEN;
    return true;
  }

  // 6. King
  const ColoredPiece kingPiece = (color == Color::WHITE) ? ColoredPiece::WHITE_KING : ColoredPiece::BLACK_KING;
  const Bitboard kings = kKingMoves[sq] & pos.pieceBitboards_[kingPiece] & colorBB & occupied;
  if (kings) {
    attackerSquare = lsb_i_promise_board_is_not_empty(kings);
    attackerPiece = Piece::KING;
    return true;
  }

  return false;
}

/**
 * Static Exchange Evaluation (SEE).
 * Simulates a sequence of captures on a single square (the destination of `move`)
 * and returns the expected material gain/loss score.
 * 
 * Score >= 0 means the capture wins material or is an equal trade.
 * Score < 0 means the capture loses material.
 */
inline int see(const Position& pos, Move move) {
  // If the move is not a capture and not a promotion, it has a strict SEE outcome of 0.
  // Exception: En passant requires special consideration due to capturing on a different square.
  if (move.moveType == MoveType::EN_PASSANT) {
    // EN PASSANT: Capturing pawn is removed, target pawn is removed on its own square.
    // Extremely complex to simulate cleanly with bitboards, so we usually just skip it or approximate.
    // Given the material balance (Pawn for Pawn), it's almost strictly equal (0) unless undefended.
    return 0; // Approximated
  }

  if (move.moveType == MoveType::CASTLE) {
    return 0; // Castling doesn't exchange material.
  }

  int gain[32] = {0};
  int depth = 0;

  // Initial target
  const SafeSquare sq = move.to;
  Piece targetPiece = cp2p(pos.tiles_[sq]);

  // Handle quiet moves (could be a promotion)
  if (targetPiece == Piece::NO_PIECE && move.moveType != MoveType::PROMOTION) {
      return 0;
  }
  
  // Starting Value (value of piece on destination square)
  gain[0] = kSeePieceValues[targetPiece];
  
  // If it's a promotion, we also gain the difference between the promoted piece and the pawn.
  if (move.moveType == MoveType::PROMOTION) {
      Piece promPiece = Piece::NO_PIECE;
      if (move.promotion == 0) promPiece = Piece::KNIGHT;
      if (move.promotion == 1) promPiece = Piece::BISHOP;
      if (move.promotion == 2) promPiece = Piece::ROOK;
      if (move.promotion == 3) promPiece = Piece::QUEEN;
      
      gain[0] += kSeePieceValues[promPiece] - kSeePieceValues[Piece::PAWN];
      
      // The promoted piece is what will be captured next!
      targetPiece = promPiece;
  }

  // We need to keep track of the piece being moved, since that's what's currently vulnerable
  Piece currentAttacker = cp2p(pos.tiles_[move.from]);
  
  // Create an occupancy bitboard and remove the moving piece so X-ray attacks can flow.
  Bitboard occupied = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];
  // Note: we don't 'move' the bit to the destination until the captures loop, we just clear its start.
  occupied &= ~bb(move.from);

  // Switch perspective to the defender (the opponent trying to recapture our piece).
  Color color = opposite_color(pos.turn_);

  while (true) {
    depth++;
    // Speculative value if the opponent successfully recaptures `currentAttacker`
    gain[depth] = kSeePieceValues[currentAttacker] - gain[depth - 1];

    // If simulating the *next* recapture is worse than stopping now, 
    // we can optimize out the rest of the branch (alpha-beta for SEE).
    if (std::max(-gain[depth - 1], gain[depth]) < 0) {
      break; 
    }

    Piece nextAttacker = Piece::NO_PIECE;
    SafeSquare nextAttackerSq = SafeSquare(0);

    if (!get_least_valuable_attacker(pos, sq, color, occupied, nextAttacker, nextAttackerSq)) {
      break;
    }

    // Found an attacker!
    // Update occupied bitboard for X-rays by removing the capturing piece's original location.
    occupied &= ~bb(nextAttackerSq);

    currentAttacker = nextAttacker;
    color = opposite_color(color);
  }

  // Minimax the scores backwards to find the forced outcome
  while (--depth > 0) {
    gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
  }

  return gain[0];
}

} // namespace ChessEngine

#endif // MOVEGEN_SEE_H
