#ifndef THREATS_H
#define THREATS_H

#include "Utils.h"
#include "Geometry.h"
#include "Position.h"

#include "movegen/bishops.h"
#include "movegen/rooks.h"
#include "movegen/knights.h"
#include "movegen/kings.h"

namespace ChessEngine {

inline Bitboard at_least_two(Bitboard a, Bitboard b) {
  return a & b;
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c) {
  return (a & b) | (a & c) | (b & c);
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d) {
  // return (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d);
  // return (a & (b | c | d)) | (b & (c | d)) | (c & d);  // 8 ops
  // return ((a ^ b) & (c ^ d)) | (a & b) | (c & d);  // 7 ops
  return ((a | b) & (c | d)) | (a & b) | (c & d);  // 7 ops
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d, Bitboard e) {
  // return (a & b) | (a & c) | (a & d) | (a & e) | (b & c) | (b & d) | (b & e) | (c & d) | (c & e) | (d & e);
  // return (a & (b | c | d | e)) | (b & (c | d | e)) | (c & (d | e)) | (d & e);  // 13 ops
  return ((a | b) & (c | d | e)) | (a & b) | at_least_two(c, d, e);  // 12 ops
}

inline Bitboard at_least_two(Bitboard a, Bitboard b, Bitboard c, Bitboard d, Bitboard e, Bitboard f) {
  return ((a | b | c) & (d | e | f)) | at_least_two(a, b, c) | at_least_two(d, e, f);  // 21 ops
}

struct Threats {
  Bitboard whitePawnTargets;
  Bitboard whiteKnightTargets;
  Bitboard whiteBishopTargets;
  Bitboard whiteRookTargets;
  Bitboard whiteQueenTargets;
  Bitboard whiteKingTargets;

  Bitboard blackPawnTargets;
  Bitboard blackKnightTargets;
  Bitboard blackBishopTargets;
  Bitboard blackRookTargets;
  Bitboard blackQueenTargets;
  Bitboard blackKingTargets;

  Bitboard whiteTargets;
  Bitboard whiteDoubleTargets;
  Bitboard blackTargets;
  Bitboard blackDoubleTargets;

  // TODO: use these.
  Bitboard badForWhite[7];
  Bitboard badForBlack[7];

  // template<ColoredPiece cp>
  // Bitboard targets() const {
  //   constexpr bool isOurColor = (cp2color(cp) == Color::WHITE);
  //   constexpr Piece piece = cp2p(cp);
  //   switch (piece) {
  //     case Piece::PAWN:
  //       return isOurColor ? whitePawnTargets : blackPawnTargets;
  //     case Piece::KNIGHT:
  //       return isOurColor ? whiteKnightTargets : blackKnightTargets;
  //     case Piece::BISHOP:
  //       return isOurColor ? whiteBishopTargets : blackBishopTargets;
  //     case Piece::ROOK:
  //       return isOurColor ? whiteRookTargets : blackRookTargets;
  //     case Piece::QUEEN:
  //       return isOurColor ? whiteQueenTargets : blackQueenTargets;
  //     case Piece::KING:
  //       return isOurColor ? whiteKingTargets : blackKingTargets;
  //     case Piece::NO_PIECE:
  //       return kEmptyBitboard;
  //   }
  // }

  template<Color US>
  Bitboard badForOur(Piece piece) const {
    if constexpr (US == Color::WHITE) {
      return badForWhite[piece];
    } else {
      return badForBlack[piece];
    }
  }

  template<ColoredPiece cp>
  Bitboard badFor() const {
    constexpr Color color = cp2color(cp);
    constexpr Piece piece = cp2p(cp);
    if (color == Color::WHITE) {
      return badForWhite[piece];
    } else {
      return badForBlack[piece];
    }
  }

  // TODO: bishops can attack one square through our own pawns.
  Threats(const Position& pos) {
    constexpr Direction kForward = Direction::NORTH;
    constexpr Direction kForwardRight = (kForward == Direction::NORTH ? Direction::NORTH_EAST : Direction::SOUTH_WEST);
    constexpr Direction kForwardLeft = (kForward == Direction::NORTH ? Direction::NORTH_WEST : Direction::SOUTH_EAST);
    constexpr Direction kBackwardRight = (kForward == Direction::NORTH ? Direction::SOUTH_WEST : Direction::NORTH_EAST);
    constexpr Direction kBackwardLeft = (kForward == Direction::NORTH ? Direction::SOUTH_EAST : Direction::NORTH_WEST);

    const SafeSquare ourKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::KING>()]);
    const SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::KING>()]);

    const Bitboard everyone = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];

    const Bitboard ourRooklikePieces = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::QUEEN>()];
    const Bitboard theirRooklikePieces = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::ROOK>()] | pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::QUEEN>()];
    const Bitboard ourBishoplikePieces = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::QUEEN>()];
    const Bitboard theirBishoplikePieces = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::BISHOP>()] | pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::QUEEN>()];

    Bitboard ourPawn1 = shift<kForwardRight>(pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::PAWN>()]);
    Bitboard ourPawn2 = shift<kForwardLeft>(pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::PAWN>()]);
    Bitboard theirPawn1 = shift<kBackwardRight>(pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::PAWN>()]);
    Bitboard theirPawn2 = shift<kBackwardLeft>(pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::PAWN>()]);

    const Bitboard ourKnights = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::KNIGHT>()];
    const Bitboard ourBishops = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::BISHOP>()];
    const Bitboard ourRooks = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::ROOK>()];
    const Bitboard ourQueens = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::QUEEN>()];
    const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<Color::WHITE, Piece::KING>()];

    const Bitboard theirKnights = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::KNIGHT>()];
    const Bitboard theirBishops = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::BISHOP>()];
    const Bitboard theirRooks = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::ROOK>()];
    const Bitboard theirQueens = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::QUEEN>()];
    const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<Color::BLACK, Piece::KING>()];

    // In general we assume "sane" positions -- no more than 2 knights, 2 bishops, 2 rooks, 1 queen (on each side).

    Bitboard ourKnight1_moves = kKnightMoves[lsb_or_none(ourKnights)];
    Bitboard ourKnight2_moves = kKnightMoves[msb_or_none(ourKnights)] * (std::popcount(ourKnights) > 1);
    Bitboard theirKnight1_moves = kKnightMoves[lsb_or_none(theirKnights)];
    Bitboard theirKnight2_moves = kKnightMoves[msb_or_none(theirKnights)] * (std::popcount(theirKnights) > 1);

    // Hard choice: do we include ourQueens in occupied?
    // YES: X is not typically threatend by "2 pieces" in this scenario, since X is usually defended by something less valuable than a queen [B Q X]
    //  NO: The case where we *do* want to count this as "2 attackers" is in attacks on the king, which are very important!

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

    this->whitePawnTargets = ourPawn1 | ourPawn2;
    this->blackPawnTargets = theirPawn1 | theirPawn2;
    this->whiteKnightTargets = ourKnight1_moves | ourKnight2_moves;
    this->blackKnightTargets = theirKnight1_moves | theirKnight2_moves;
    this->whiteBishopTargets = ourBishops1_moves | ourBishops2_moves;
    this->blackBishopTargets = theirBishops1_moves | theirBishops2_moves;
    this->whiteRookTargets = ourRooks1_moves | ourRooks2_moves;
    this->blackRookTargets = theirRooks1_moves | theirRooks2_moves;
    this->whiteQueenTargets = whiteQueenTargets;
    this->blackQueenTargets = whiteQueenTargets;
    this->whiteKingTargets = whiteKingTargets;
    this->blackKingTargets = blackKingTargets;

    this->whiteTargets = this->whitePawnTargets;
    this->blackTargets = this->blackPawnTargets;
    this->whiteDoubleTargets = ourPawn1 & ourPawn2;
    this->blackDoubleTargets = theirPawn1 & theirPawn2;

    this->whiteDoubleTargets |= ourKnight1_moves & this->whiteTargets;
    this->whiteTargets |= ourKnight1_moves;
    this->whiteDoubleTargets |= ourKnight2_moves & this->whiteTargets;
    this->whiteTargets |= ourKnight2_moves;

    this->blackDoubleTargets |= theirKnight1_moves & this->blackTargets;
    this->blackTargets |= theirKnight1_moves;
    this->blackDoubleTargets |= theirKnight2_moves & this->blackTargets;
    this->blackTargets |= theirKnight2_moves;

    // Can speed this up by assuming bishops are on opposite colors.
    this->whiteDoubleTargets |= this->whiteBishopTargets & this->whiteTargets;
    this->whiteTargets |= this->whiteBishopTargets;
    this->blackDoubleTargets |= (theirBishops1_moves | theirBishops2_moves) & this->blackTargets;
    this->blackTargets |= theirBishops1_moves | theirBishops2_moves;

    this->whiteDoubleTargets |= ourRooks1_moves & this->whiteTargets;
    this->whiteTargets |= ourRooks1_moves;
    this->whiteDoubleTargets |= ourRooks2_moves & this->whiteTargets;
    this->whiteTargets |= ourRooks2_moves;

    this->blackDoubleTargets |= theirRooks1_moves & this->blackTargets;
    this->blackTargets |= theirRooks1_moves;
    this->blackDoubleTargets |= theirRooks2_moves & this->blackTargets;
    this->blackTargets |= theirRooks2_moves;

    this->whiteDoubleTargets |= whiteQueenTargets & this->whiteTargets;
    this->whiteTargets |= whiteQueenTargets;
    this->blackDoubleTargets |= blackQueenTargets & this->blackTargets;
    this->blackTargets |= blackQueenTargets;

    this->whiteDoubleTargets |= whiteKingTargets & this->whiteTargets;
    this->whiteTargets |= whiteKingTargets;
    this->blackDoubleTargets |= blackKingTargets & this->blackTargets;
    this->blackTargets |= blackKingTargets;

    const Bitboard badForAllOfUs = this->blackTargets & ~this->whiteTargets;
    const Bitboard badForAllOfThem = this->whiteTargets & ~this->blackTargets;

    this->badForWhite[Piece::PAWN] = badForAllOfUs;
    this->badForBlack[Piece::PAWN] = badForAllOfThem;

    // Not defended by a pawn and attacked more than once.
    this->badForWhite[Piece::PAWN] |= (~(ourPawn1 | ourPawn2)) & (this->blackDoubleTargets & ~this->whiteDoubleTargets);
    this->badForBlack[Piece::PAWN] |= (~(theirPawn1 | theirPawn2)) & (this->whiteDoubleTargets & ~this->blackDoubleTargets);

    // Defended by one pawn and attacked by a pawn and a piece.
    this->badForWhite[Piece::PAWN] |= (ourPawn1 ^ ourPawn2) & (theirPawn1 | theirPawn2) & (this->blackDoubleTargets & ~this->whiteDoubleTargets);
    this->badForBlack[Piece::PAWN] |= (theirPawn1 ^ theirPawn2) & (ourPawn1 | ourPawn2) & (this->whiteDoubleTargets & ~this->blackDoubleTargets);

    this->badForWhite[Piece::PAWN] &= ~pos.colorBitboards_[Color::BLACK];
    this->badForBlack[Piece::PAWN] &= ~pos.colorBitboards_[Color::WHITE];

    // We exclude the piece being considered as a valid defender, since "badForWhite" is frequently used to answer
    // the question "is it okay to move here?", and you can't keep defending a square if you move to it!

    // When computing double attacks, pieces that are more valuable than the piece being considered are merged
    // into a single attack -- if your bishop is attacked by two knights, it doesn't really matter if it is
    // defended by a king, a queen, and a rook -- it's still hanging!

    Bitboard ourTwo_def, ourTwo_atk, theirTwo_def, theirTwo_atk;
    Bitboard ourOne_def, ourOne_atk, theirOne_def, theirOne_atk;

    const Bitboard whiteMinorTargets = this->whiteKnightTargets | this->whiteBishopTargets;
    const Bitboard blackMinorTargets = this->blackKnightTargets | this->blackBishopTargets;
    const Bitboard whiteRoyalTargets = whiteQueenTargets | whiteKingTargets;
    const Bitboard blackRoyalTargets = blackQueenTargets | blackKingTargets;

    ourOne_def = (ourKnight1_moves & ourKnight2_moves) | this->whiteBishopTargets | this->whiteRookTargets | whiteRoyalTargets;
    ourOne_atk = whiteMinorTargets | this->whiteRookTargets | whiteRoyalTargets;
    theirOne_def = (theirKnight1_moves & theirKnight2_moves) | this->blackBishopTargets | this->blackRookTargets | blackRoyalTargets;
    theirOne_atk = blackMinorTargets | this->blackRookTargets | blackRoyalTargets;
    // Minor optimization: we merge both bishops since their targets are mutually exclusive 99% of the time.
    ourTwo_def = at_least_two(
      ourKnight1_moves & ourKnight2_moves, this->whiteBishopTargets, this->whiteRookTargets | whiteRoyalTargets
    );
    ourTwo_atk = at_least_two(
      ourKnight1_moves, ourKnight2_moves, this->whiteBishopTargets, this->whiteRookTargets | whiteRoyalTargets
    );
    theirTwo_def = at_least_two(
      theirKnight1_moves & theirKnight2_moves, this->blackBishopTargets, this->blackRookTargets | blackRoyalTargets
    );
    theirTwo_atk = at_least_two(
      theirKnight1_moves, theirKnight2_moves, this->blackBishopTargets, this->blackRookTargets | blackRoyalTargets
    );
    this->badForWhite[Piece::KNIGHT] = badForAllOfUs | this->blackPawnTargets | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
    this->badForWhite[Piece::KNIGHT] &= ~(theirKings | theirQueens | theirRooks | theirKnights | theirBishops);
    this->badForBlack[Piece::KNIGHT] = badForAllOfThem | this->whitePawnTargets | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
    this->badForBlack[Piece::KNIGHT] &= ~(ourKings | ourQueens | ourRooks | ourKnights | ourBishops);

    ourOne_def = this->whiteKnightTargets | (ourBishops1_moves & ourBishops2_moves) | this->whiteRookTargets | whiteRoyalTargets;
    ourOne_atk = whiteMinorTargets | this->whiteRookTargets | whiteRoyalTargets;
    theirOne_def = theirKnight1_moves | theirKnight2_moves | (theirBishops1_moves & theirBishops2_moves) | this->blackRookTargets | blackRoyalTargets;
    theirOne_atk = blackMinorTargets | this->blackRookTargets | blackRoyalTargets;
    ourTwo_def = at_least_two(ourKnight1_moves, ourKnight2_moves, this->whiteRookTargets | whiteRoyalTargets);
    ourTwo_atk = at_least_two(ourKnight1_moves, ourKnight2_moves, this->whiteBishopTargets, this->whiteRookTargets | whiteRoyalTargets);
    theirTwo_def = at_least_two(theirKnight1_moves, theirKnight2_moves, this->blackRookTargets | blackRoyalTargets);
    theirTwo_atk = at_least_two(theirKnight1_moves, theirKnight2_moves, this->blackBishopTargets, this->blackRookTargets | blackRoyalTargets);
    this->badForWhite[Piece::BISHOP] = badForAllOfUs | this->blackPawnTargets | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
    this->badForWhite[Piece::BISHOP] &= ~(theirKings | theirQueens | theirRooks | theirKnights | theirBishops);
    this->badForBlack[Piece::BISHOP] = badForAllOfThem | this->whitePawnTargets | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
    this->badForBlack[Piece::BISHOP] &= ~(ourKings | ourQueens | ourRooks | ourKnights | ourBishops);

    ourOne_def = whiteMinorTargets | (ourRooks1_moves & ourRooks2_moves) | whiteRoyalTargets;
    ourOne_atk = whiteMinorTargets | this->whiteRookTargets | whiteRoyalTargets;
    theirOne_def = blackMinorTargets | (theirRooks1_moves & theirRooks2_moves) | blackRoyalTargets;
    theirOne_atk = blackMinorTargets | this->blackRookTargets | blackRoyalTargets;
    ourTwo_def = at_least_two(ourKnight1_moves, ourKnight2_moves, this->whiteBishopTargets, ourRooks1_moves & ourRooks2_moves, whiteRoyalTargets);
    ourTwo_atk = at_least_two(ourKnight1_moves, ourKnight2_moves, this->whiteBishopTargets, ourRooks1_moves, ourRooks2_moves, whiteRoyalTargets);
    theirTwo_def = at_least_two(theirKnight1_moves, theirKnight2_moves, this->blackBishopTargets, theirRooks1_moves & theirRooks2_moves, blackRoyalTargets);
    theirTwo_atk = at_least_two(theirKnight1_moves, theirKnight2_moves, this->blackBishopTargets, theirRooks1_moves, theirRooks2_moves, blackRoyalTargets);
    this->badForWhite[Piece::ROOK] = badForAllOfUs | (this->blackPawnTargets | blackMinorTargets) | (theirTwo_atk & ~ourTwo_def) | (theirOne_atk & ~ourOne_def);
    this->badForWhite[Piece::ROOK] &= ~(theirKings | theirQueens | theirRooks);
    this->badForBlack[Piece::ROOK] = badForAllOfThem | (this->whitePawnTargets | whiteMinorTargets) | (ourTwo_atk & ~theirTwo_def) | (ourOne_atk & ~theirOne_def);
    this->badForBlack[Piece::ROOK] &= ~(ourKings | ourQueens | ourRooks);

    this->badForWhite[Piece::QUEEN] = badForAllOfUs | this->blackPawnTargets | blackMinorTargets | this->blackRookTargets | ((blackRoyalTargets) & ~this->whiteTargets);
    this->badForWhite[Piece::QUEEN] &= ~(theirKings | theirQueens);
    this->badForBlack[Piece::QUEEN] = badForAllOfThem | this->whitePawnTargets | whiteMinorTargets | this->whiteRookTargets | ((whiteRoyalTargets) & ~this->blackTargets);
    this->badForBlack[Piece::QUEEN] &= ~(ourKings | ourQueens);

    this->badForWhite[Piece::KING] = this->blackTargets;
    this->badForBlack[Piece::KING] = this->whiteTargets;
  }
};

}  // namespace ChessEngine

#endif  // THREATS_H