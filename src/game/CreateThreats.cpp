#ifndef CREATE_THREATS_H
#define CREATE_THREATS_H

#include "Threats.h"

#include "movegen/bishops.h"
#include "movegen/rooks.h"
#include "movegen/knights.h"
#include "movegen/kings.h"

namespace ChessEngine {

// TODO: bishops can attack one square through our own pawns.
void create_threats(const TypeSafeArray<Bitboard, kNumColoredPieces, ColoredPiece>& pieceBitboards, const TypeSafeArray<Bitboard, Color::NUM_COLORS, Color>& colorBitboards, Threats *out) {
  constexpr Direction kForward = Direction::NORTH;
  constexpr Direction kForwardRight = (kForward == Direction::NORTH ? Direction::NORTH_EAST : Direction::SOUTH_WEST);
  constexpr Direction kForwardLeft = (kForward == Direction::NORTH ? Direction::NORTH_WEST : Direction::SOUTH_EAST);
  constexpr Direction kBackwardRight = (kForward == Direction::NORTH ? Direction::SOUTH_WEST : Direction::NORTH_EAST);
  constexpr Direction kBackwardLeft = (kForward == Direction::NORTH ? Direction::SOUTH_EAST : Direction::NORTH_WEST);

  const SafeSquare ourKingSq = lsb_i_promise_board_is_not_empty(pieceBitboards[coloredPiece<Color::WHITE, Piece::KING>()]);
  const SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(pieceBitboards[coloredPiece<Color::BLACK, Piece::KING>()]);

  const Bitboard everyone = colorBitboards[Color::WHITE] | colorBitboards[Color::BLACK];

  const Bitboard ourRooklikePieces = pieceBitboards[coloredPiece<Color::WHITE, Piece::ROOK>()] | pieceBitboards[coloredPiece<Color::WHITE, Piece::QUEEN>()];
  const Bitboard theirRooklikePieces = pieceBitboards[coloredPiece<Color::BLACK, Piece::ROOK>()] | pieceBitboards[coloredPiece<Color::BLACK, Piece::QUEEN>()];
  const Bitboard ourBishoplikePieces = pieceBitboards[coloredPiece<Color::WHITE, Piece::BISHOP>()] | pieceBitboards[coloredPiece<Color::WHITE, Piece::QUEEN>()];
  const Bitboard theirBishoplikePieces = pieceBitboards[coloredPiece<Color::BLACK, Piece::BISHOP>()] | pieceBitboards[coloredPiece<Color::BLACK, Piece::QUEEN>()];

  Bitboard ourPawn1 = shift<kForwardRight>(pieceBitboards[coloredPiece<Color::WHITE, Piece::PAWN>()]);
  Bitboard ourPawn2 = shift<kForwardLeft>(pieceBitboards[coloredPiece<Color::WHITE, Piece::PAWN>()]);
  Bitboard theirPawn1 = shift<kBackwardRight>(pieceBitboards[coloredPiece<Color::BLACK, Piece::PAWN>()]);
  Bitboard theirPawn2 = shift<kBackwardLeft>(pieceBitboards[coloredPiece<Color::BLACK, Piece::PAWN>()]);

  const Bitboard ourKnights = pieceBitboards[coloredPiece<Color::WHITE, Piece::KNIGHT>()];
  const Bitboard ourBishops = pieceBitboards[coloredPiece<Color::WHITE, Piece::BISHOP>()];
  const Bitboard ourRooks = pieceBitboards[coloredPiece<Color::WHITE, Piece::ROOK>()];
  const Bitboard ourQueens = pieceBitboards[coloredPiece<Color::WHITE, Piece::QUEEN>()];
  const Bitboard ourKings = pieceBitboards[coloredPiece<Color::WHITE, Piece::KING>()];

  const Bitboard theirKnights = pieceBitboards[coloredPiece<Color::BLACK, Piece::KNIGHT>()];
  const Bitboard theirBishops = pieceBitboards[coloredPiece<Color::BLACK, Piece::BISHOP>()];
  const Bitboard theirRooks = pieceBitboards[coloredPiece<Color::BLACK, Piece::ROOK>()];
  const Bitboard theirQueens = pieceBitboards[coloredPiece<Color::BLACK, Piece::QUEEN>()];
  const Bitboard theirKings = pieceBitboards[coloredPiece<Color::BLACK, Piece::KING>()];

  // In general we assume "sane" positions -- no more than 2 knights, 2 bishops, 2 rooks, 1 queen (on each side).

  Bitboard ourKnight1_moves = kKnightMoves[lsb_or_none(ourKnights)];
  Bitboard ourKnight2_moves = kKnightMoves[msb_or_none(ourKnights)] * (std::popcount(ourKnights) > 1);
  Bitboard theirKnight1_moves = kKnightMoves[lsb_or_none(theirKnights)];
  Bitboard theirKnight2_moves = kKnightMoves[msb_or_none(theirKnights)] * (std::popcount(theirKnights) > 1);

  // Hard choice: do we include ourQueens in occupied?
  // YES: X is not typically threatend by "2 pieces" in out scenario, since X is usually defended by something less valuable than a queen [B Q X]
  //  NO: The case where we *do* want to count out as "2 attackers" is in attacks on the king, which are very important!

  // TODO: get rid of branches for sliding pieces (due to "sliding_moves" in "sliding.h")

  Bitboard ourBishops1_moves = ourBishops == kEmptyBitboard ? kEmptyBitboard : compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(ourBishops), everyone & ~ourBishops);
  Bitboard ourBishops2_moves = std::popcount(ourBishops) < 2 ? kEmptyBitboard : compute_one_bishops_targets(msb_i_promise_board_is_not_empty(ourBishops), everyone & ~ourBishops);
  Bitboard theirBishops1_moves = theirBishops == kEmptyBitboard ? kEmptyBitboard : compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(theirBishops), everyone & ~theirBishops);
  Bitboard theirBishops2_moves = std::popcount(theirBishops) < 2 ? kEmptyBitboard : compute_one_bishops_targets(msb_i_promise_board_is_not_empty(theirBishops), everyone & ~theirBishops);

  Bitboard ourRooks1_moves = ourRooks == kEmptyBitboard ? kEmptyBitboard : compute_single_rook_moves(lsb_i_promise_board_is_not_empty(ourRooks), everyone & ~ourRooks);
  Bitboard ourRooks2_moves = std::popcount(ourRooks) < 2 ? kEmptyBitboard : compute_single_rook_moves(msb_i_promise_board_is_not_empty(ourRooks), everyone & ~ourRooks);
  Bitboard theirRooks1_moves = theirRooks == kEmptyBitboard ? kEmptyBitboard : compute_single_rook_moves(lsb_i_promise_board_is_not_empty(theirRooks), everyone & ~theirRooks);
  Bitboard theirRooks2_moves = std::popcount(theirRooks) < 2 ? kEmptyBitboard : compute_single_rook_moves(msb_i_promise_board_is_not_empty(theirRooks), everyone & ~theirRooks);

  Bitboard whiteQueenTargets = kEmptyBitboard;
  if (ourQueens != kEmptyBitboard) {
    whiteQueenTargets |= compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(ourQueens), everyone & ~ourBishops);
    whiteQueenTargets |= compute_single_rook_moves(lsb_i_promise_board_is_not_empty(ourQueens), everyone & ~ourRooks);
  }
  Bitboard blackQueenTargets = kEmptyBitboard;
  if (theirQueens != kEmptyBitboard) {
    blackQueenTargets |= compute_one_bishops_targets(lsb_i_promise_board_is_not_empty(theirQueens), everyone & ~theirBishops & ~ourBishops);
    blackQueenTargets |= compute_single_rook_moves(lsb_i_promise_board_is_not_empty(theirQueens), everyone & ~theirRooks & ~ourRooks);
  }

  Bitboard whiteKingTargets = kKingMoves[ourKingSq];
  Bitboard blackKingTargets = kKingMoves[theirKingSq];

  out->whitePawnTargets = ourPawn1 | ourPawn2;
  out->blackPawnTargets = theirPawn1 | theirPawn2;
  out->whiteKnightTargets = ourKnight1_moves | ourKnight2_moves;
  out->blackKnightTargets = theirKnight1_moves | theirKnight2_moves;
  out->whiteBishopTargets = ourBishops1_moves | ourBishops2_moves;
  out->blackBishopTargets = theirBishops1_moves | theirBishops2_moves;
  out->whiteRookTargets = ourRooks1_moves | ourRooks2_moves;
  out->blackRookTargets = theirRooks1_moves | theirRooks2_moves;
  out->whiteQueenTargets = whiteQueenTargets;
  out->blackQueenTargets = whiteQueenTargets;
  out->whiteKingTargets = whiteKingTargets;
  out->blackKingTargets = blackKingTargets;

  out->whiteTargets = out->whitePawnTargets;
  out->blackTargets = out->blackPawnTargets;
  out->whiteDoubleTargets = ourPawn1 & ourPawn2;
  out->blackDoubleTargets = theirPawn1 & theirPawn2;

  out->whiteDoubleTargets |= ourKnight1_moves & out->whiteTargets;
  out->whiteTargets |= ourKnight1_moves;
  out->whiteDoubleTargets |= ourKnight2_moves & out->whiteTargets;
  out->whiteTargets |= ourKnight2_moves;

  out->blackDoubleTargets |= theirKnight1_moves & out->blackTargets;
  out->blackTargets |= theirKnight1_moves;
  out->blackDoubleTargets |= theirKnight2_moves & out->blackTargets;
  out->blackTargets |= theirKnight2_moves;

  // Can speed out up by assuming bishops are on opposite colors.
  out->whiteDoubleTargets |= out->whiteBishopTargets & out->whiteTargets;
  out->whiteTargets |= out->whiteBishopTargets;
  out->blackDoubleTargets |= (theirBishops1_moves | theirBishops2_moves) & out->blackTargets;
  out->blackTargets |= theirBishops1_moves | theirBishops2_moves;

  out->whiteDoubleTargets |= ourRooks1_moves & out->whiteTargets;
  out->whiteTargets |= ourRooks1_moves;
  out->whiteDoubleTargets |= ourRooks2_moves & out->whiteTargets;
  out->whiteTargets |= ourRooks2_moves;

  out->blackDoubleTargets |= theirRooks1_moves & out->blackTargets;
  out->blackTargets |= theirRooks1_moves;
  out->blackDoubleTargets |= theirRooks2_moves & out->blackTargets;
  out->blackTargets |= theirRooks2_moves;

  out->whiteDoubleTargets |= whiteQueenTargets & out->whiteTargets;
  out->whiteTargets |= whiteQueenTargets;
  out->blackDoubleTargets |= blackQueenTargets & out->blackTargets;
  out->blackTargets |= blackQueenTargets;

  out->whiteDoubleTargets |= whiteKingTargets & out->whiteTargets;
  out->whiteTargets |= whiteKingTargets;
  out->blackDoubleTargets |= blackKingTargets & out->blackTargets;
  out->blackTargets |= blackKingTargets;

  const Bitboard badForAllOfUs = out->blackTargets & ~out->whiteTargets;
  const Bitboard badForAllOfThem = out->whiteTargets & ~out->blackTargets;

  out->badForWhite[Piece::PAWN] = badForAllOfUs;
  out->badForBlack[Piece::PAWN] = badForAllOfThem;

  // Not defended by a pawn and attacked more than once.
  out->badForWhite[Piece::PAWN] |= (~(ourPawn1 | ourPawn2)) & (out->blackDoubleTargets & ~out->whiteDoubleTargets);
  out->badForBlack[Piece::PAWN] |= (~(theirPawn1 | theirPawn2)) & (out->whiteDoubleTargets & ~out->blackDoubleTargets);

  // Defended by one pawn and attacked by a pawn and a piece.
  out->badForWhite[Piece::PAWN] |= (ourPawn1 ^ ourPawn2) & (theirPawn1 | theirPawn2) & (out->blackDoubleTargets & ~out->whiteDoubleTargets);
  out->badForBlack[Piece::PAWN] |= (theirPawn1 ^ theirPawn2) & (ourPawn1 | ourPawn2) & (out->whiteDoubleTargets & ~out->blackDoubleTargets);

  out->badForWhite[Piece::PAWN] &= ~colorBitboards[Color::BLACK];
  out->badForBlack[Piece::PAWN] &= ~colorBitboards[Color::WHITE];

  // We exclude the piece being considered as a valid defender, since "badForWhite" is frequently used to answer
  // the question "is it okay to move here?", and you can't keep defending a square if you move to it!

  // When computing double attacks, pieces that are more valuable than the piece being considered are merged
  // into a single attack -- if your bishop is attacked by two knights, it doesn't really matter if it is
  // defended by a king, a queen, and a rook -- it's still hanging!

  Bitboard ourTwo_def, ourTwo_atk, theirTwo_def, theirTwo_atk;
  Bitboard ourOne_def, ourOne_atk, theirOne_def, theirOne_atk;

  const Bitboard whiteMinorTargets = out->whiteKnightTargets | out->whiteBishopTargets;
  const Bitboard blackMinorTargets = out->blackKnightTargets | out->blackBishopTargets;
  const Bitboard whiteRoyalTargets = whiteQueenTargets | whiteKingTargets;
  const Bitboard blackRoyalTargets = blackQueenTargets | blackKingTargets;

  ourOne_def = (ourKnight1_moves & ourKnight2_moves) | out->whiteBishopTargets | out->whiteRookTargets | whiteRoyalTargets;
  ourOne_atk = whiteMinorTargets | out->whiteRookTargets | whiteRoyalTargets;
  theirOne_def = (theirKnight1_moves & theirKnight2_moves) | out->blackBishopTargets | out->blackRookTargets | blackRoyalTargets;
  theirOne_atk = blackMinorTargets | out->blackRookTargets | blackRoyalTargets;
  // Minor optimization: we merge both bishops since their targets are mutually exclusive 99% of the time.
  ourTwo_def = at_least_two(
    ourKnight1_moves & ourKnight2_moves, out->whiteBishopTargets, out->whiteRookTargets | whiteRoyalTargets
  );
  ourTwo_atk = at_least_two(
    ourKnight1_moves, ourKnight2_moves, out->whiteBishopTargets, out->whiteRookTargets | whiteRoyalTargets
  );
  theirTwo_def = at_least_two(
    theirKnight1_moves & theirKnight2_moves, out->blackBishopTargets, out->blackRookTargets | blackRoyalTargets
  );
  theirTwo_atk = at_least_two(
    theirKnight1_moves, theirKnight2_moves, out->blackBishopTargets, out->blackRookTargets | blackRoyalTargets
  );
  out->badForWhite[Piece::KNIGHT] = badForAllOfUs | out->blackPawnTargets | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
  out->badForWhite[Piece::KNIGHT] &= ~(theirKings | theirQueens | theirRooks | theirKnights | theirBishops);
  out->badForBlack[Piece::KNIGHT] = badForAllOfThem | out->whitePawnTargets | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
  out->badForBlack[Piece::KNIGHT] &= ~(ourKings | ourQueens | ourRooks | ourKnights | ourBishops);

  ourOne_def = out->whiteKnightTargets | (ourBishops1_moves & ourBishops2_moves) | out->whiteRookTargets | whiteRoyalTargets;
  ourOne_atk = whiteMinorTargets | out->whiteRookTargets | whiteRoyalTargets;
  theirOne_def = theirKnight1_moves | theirKnight2_moves | (theirBishops1_moves & theirBishops2_moves) | out->blackRookTargets | blackRoyalTargets;
  theirOne_atk = blackMinorTargets | out->blackRookTargets | blackRoyalTargets;
  ourTwo_def = at_least_two(ourKnight1_moves, ourKnight2_moves, out->whiteRookTargets | whiteRoyalTargets);
  ourTwo_atk = at_least_two(ourKnight1_moves, ourKnight2_moves, out->whiteBishopTargets, out->whiteRookTargets | whiteRoyalTargets);
  theirTwo_def = at_least_two(theirKnight1_moves, theirKnight2_moves, out->blackRookTargets | blackRoyalTargets);
  theirTwo_atk = at_least_two(theirKnight1_moves, theirKnight2_moves, out->blackBishopTargets, out->blackRookTargets | blackRoyalTargets);
  out->badForWhite[Piece::BISHOP] = badForAllOfUs | out->blackPawnTargets | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
  out->badForWhite[Piece::BISHOP] &= ~(theirKings | theirQueens | theirRooks | theirKnights | theirBishops);
  out->badForBlack[Piece::BISHOP] = badForAllOfThem | out->whitePawnTargets | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
  out->badForBlack[Piece::BISHOP] &= ~(ourKings | ourQueens | ourRooks | ourKnights | ourBishops);

  ourOne_def = whiteMinorTargets | (ourRooks1_moves & ourRooks2_moves) | whiteRoyalTargets;
  ourOne_atk = whiteMinorTargets | out->whiteRookTargets | whiteRoyalTargets;
  theirOne_def = blackMinorTargets | (theirRooks1_moves & theirRooks2_moves) | blackRoyalTargets;
  theirOne_atk = blackMinorTargets | out->blackRookTargets | blackRoyalTargets;
  ourTwo_def = at_least_two(ourKnight1_moves, ourKnight2_moves, out->whiteBishopTargets, ourRooks1_moves & ourRooks2_moves, whiteRoyalTargets);
  ourTwo_atk = at_least_two(ourKnight1_moves, ourKnight2_moves, out->whiteBishopTargets, ourRooks1_moves, ourRooks2_moves, whiteRoyalTargets);
  theirTwo_def = at_least_two(theirKnight1_moves, theirKnight2_moves, out->blackBishopTargets, theirRooks1_moves & theirRooks2_moves, blackRoyalTargets);
  theirTwo_atk = at_least_two(theirKnight1_moves, theirKnight2_moves, out->blackBishopTargets, theirRooks1_moves, theirRooks2_moves, blackRoyalTargets);
  out->badForWhite[Piece::ROOK] = badForAllOfUs | (out->blackPawnTargets | blackMinorTargets) | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
  out->badForWhite[Piece::ROOK] &= ~(theirKings | theirQueens | theirRooks);
  out->badForBlack[Piece::ROOK] = badForAllOfThem | (out->whitePawnTargets | whiteMinorTargets) | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
  out->badForBlack[Piece::ROOK] &= ~(ourKings | ourQueens | ourRooks);

  out->badForWhite[Piece::QUEEN] = badForAllOfUs | out->blackPawnTargets | blackMinorTargets | out->blackRookTargets | ((blackRoyalTargets) & ~out->whiteTargets);
  out->badForWhite[Piece::QUEEN] &= ~(theirKings | theirQueens);
  out->badForBlack[Piece::QUEEN] = badForAllOfThem | out->whitePawnTargets | whiteMinorTargets | out->whiteRookTargets | ((whiteRoyalTargets) & ~out->blackTargets);
  out->badForBlack[Piece::QUEEN] &= ~(ourKings | ourQueens);

  out->badForWhite[Piece::KING] = out->blackTargets;
  out->badForBlack[Piece::KING] = out->whiteTargets;
}

}  // namespace ChessEngine

#endif  // CREATE_THREATS_H