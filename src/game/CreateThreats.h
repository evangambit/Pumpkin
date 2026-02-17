#ifndef CREATE_THREATS_H
#define CREATE_THREATS_H

#include "Threats.h"

namespace ChessEngine {

void create_threats(const TypeSafeArray<Bitboard, kNumColoredPieces, ColoredPiece>& pieceBitboards, const TypeSafeArray<Bitboard, Color::NUM_COLORS, Color>& colorBitboards, Threats *out);

}  // namespace ChessEngine

#endif  // CREATE_THREATS_H