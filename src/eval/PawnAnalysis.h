#ifndef QST_PAWN_ANALYSIS_H
#define QST_PAWN_ANALYSIS_H

#include "../game/Position.h"
#include "../game/movegen/pawns.h"

namespace ChessEngine {

template<Color US>
struct PawnAnalysis {
  Bitboard ourPassedPawns, theirPassedPawns;
  Bitboard ourIsolatedPawns, theirIsolatedPawns;
  Bitboard ourDoubledPawns, theirDoubledPawns;
  Bitboard filesWithoutOurPawns, filesWithoutTheirPawns;
  Bitboard ourOutposts, theirOutposts;

  PawnAnalysis(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
    constexpr Direction kBackward = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;

    Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    
    Bitboard aheadOfOurPawns = shift<kForward>(US == Color::WHITE ? northFill(ourPawns) : southFill(ourPawns));
    Bitboard aheadOfTheirPawns = shift<kBackward>(US == Color::WHITE ? southFill(theirPawns) : northFill(theirPawns));
    Bitboard filesWithOurPawns = US == Color::WHITE ? southFill(aheadOfOurPawns) : northFill(aheadOfOurPawns);
    Bitboard filesWithTheirPawns = US == Color::WHITE ? northFill(aheadOfTheirPawns) : southFill(aheadOfTheirPawns);
    filesWithoutOurPawns = ~filesWithOurPawns;
    filesWithoutTheirPawns = ~filesWithTheirPawns;
    this->ourPassedPawns = ourPawns & ~fatten(aheadOfTheirPawns);
    this->theirPassedPawns = theirPawns & ~fatten(aheadOfOurPawns);
    this->ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
    this->theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
    this->ourDoubledPawns = ourPawns & aheadOfOurPawns;
    this->theirDoubledPawns = theirPawns & aheadOfTheirPawns;

    // An outpost is a square that is not ahead of an enemy pawn on its left or right side,
    // but *is* protected by a friendly pawn.
    const Bitboard unsafeFromThem = shift<Direction::EAST>(aheadOfTheirPawns) | shift<Direction::WEST>(aheadOfTheirPawns);
    const Bitboard unsafeFromUs = shift<Direction::EAST>(aheadOfOurPawns) | shift<Direction::WEST>(aheadOfOurPawns);
    this->ourOutposts = compute_pawn_targets<US>(pos) & ~unsafeFromThem;
    this->theirOutposts = compute_pawn_targets<THEM>(pos) & ~unsafeFromUs;
  }
};

}  // namespace ChessEngine

#endif  // QST_PAWN_ANALYSIS_H