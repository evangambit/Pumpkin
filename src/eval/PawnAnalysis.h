#ifndef QST_PAWN_ANALYSIS_H
#define QST_PAWN_ANALYSIS_H

#include "../game/Position.h"

namespace ChessEngine {

template<Color US>
struct PawnAnalysis {
  Bitboard ourPassedPawns, theirPassedPawns;
  Bitboard ourIsolatedPawns, theirIsolatedPawns;
  Bitboard ourDoubledPawns, theirDoubledPawns;
  Bitboard filesWithoutOurPawns, filesWithoutTheirPawns;

  PawnAnalysis(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Direction kForward = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
    constexpr Direction kBackward = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;

    Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    
    Bitboard aheadOfOurPawns = US == Color::WHITE ? northFill(ourPawns) : southFill(ourPawns);
    Bitboard aheadOfTheirPawns = US == Color::WHITE ? southFill(theirPawns) : northFill(theirPawns);
    Bitboard filesWithOurPawns = US == Color::WHITE ? southFill(aheadOfOurPawns) : northFill(aheadOfOurPawns);
    Bitboard filesWithTheirPawns = US == Color::WHITE ? northFill(aheadOfTheirPawns) : southFill(aheadOfTheirPawns);
    filesWithoutOurPawns = ~filesWithOurPawns;
    filesWithoutTheirPawns = ~filesWithTheirPawns;
    this->ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(aheadOfTheirPawns));
    this->theirPassedPawns = theirPawns & ~shift<kForward>(fatten(aheadOfOurPawns));
    this->ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
    this->theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
    this->ourDoubledPawns = ourPawns & shift<kForward>(aheadOfOurPawns);
    this->theirDoubledPawns = theirPawns & shift<kBackward>(aheadOfTheirPawns);
  }
};

}  // namespace ChessEngine

#endif  // QST_PAWN_ANALYSIS_H