#ifndef SRC_EVAL_NNUE_NNUEFEATUREBITMAPTYPE_H
#define SRC_EVAL_NNUE_NNUEFEATUREBITMAPTYPE_H
#include "../../game/Geometry.h"
#include <string>
#include <iostream>

namespace NNUE {

enum NnueFeatureBitmapType {
  NF_WHITE_PAWN,
  NF_WHITE_KNIGHT,
  NF_WHITE_BISHOP,
  NF_WHITE_ROOK,
  NF_WHITE_QUEEN,
  NF_WHITE_KING,
  NF_WHITE_HANGING_PAWNS,
  NF_WHITE_HANGING_KNIGHTS,
  NF_WHITE_HANGING_BISHOPS,
  NF_WHITE_HANGING_ROOKS,
  NF_WHITE_HANGING_QUEENS,
  NF_WHITE_HANGING_KINGS,
  NF_BLACK_PAWN,
  NF_BLACK_KNIGHT,
  NF_BLACK_BISHOP,
  NF_BLACK_ROOK,
  NF_BLACK_QUEEN,
  NF_BLACK_KING,
  NF_BLACK_HANGING_PAWNS,
  NF_BLACK_HANGING_KNIGHTS,
  NF_BLACK_HANGING_BISHOPS,
  NF_BLACK_HANGING_ROOKS,
  NF_BLACK_HANGING_QUEENS,
  NF_BLACK_HANGING_KINGS,
  NF_COUNT
};
static_assert(NF_COUNT % 2 == 0, "NF_COUNT must be even");
static_assert(NF_COUNT / 2 == NF_BLACK_PAWN, "Half of the features must be for black pieces and half for white pieces");

// It is impossible for a pawn to be on the first or last rank, so we can
// use these indices to encode other things.

// We use some unused pawn positions to encode castling rights.
// We choose squares so that the same flipping logic that applies to pieces
// also applies to castling rights.
enum SpecialFeatures {
  WHITE_KINGSIDE_CASTLING_RIGHT = NF_WHITE_PAWN * 64 + 0,  // "white pawn on a8"
  WHITE_QUEENSIDE_CASTLING_RIGHT = NF_WHITE_PAWN * 64 + 1,

  NO_WHITE_PAWNS_A_FILE = NF_WHITE_PAWN * 64 + 56,
  NO_WHITE_PAWNS_B_FILE = NF_WHITE_PAWN * 64 + 57,
  NO_WHITE_PAWNS_C_FILE = NF_WHITE_PAWN * 64 + 58,
  NO_WHITE_PAWNS_D_FILE = NF_WHITE_PAWN * 64 + 59,
  NO_WHITE_PAWNS_E_FILE = NF_WHITE_PAWN * 64 + 60,
  NO_WHITE_PAWNS_F_FILE = NF_WHITE_PAWN * 64 + 61,
  NO_WHITE_PAWNS_G_FILE = NF_WHITE_PAWN * 64 + 62,
  NO_WHITE_PAWNS_H_FILE = NF_WHITE_PAWN * 64 + 63,

  BLACK_KINGSIDE_CASTLING_RIGHT = NF_BLACK_PAWN * 64 + 56,  // "black pawn on a1" (vertically flipped vs white's castling right)
  BLACK_QUEENSIDE_CASTLING_RIGHT = NF_BLACK_PAWN * 64 + 57,

  NO_BLACK_PAWNS_A_FILE = NF_BLACK_PAWN * 64 + 0,  // flipped vertically vs white's no pawns on a file
  NO_BLACK_PAWNS_B_FILE = NF_BLACK_PAWN * 64 + 1,
  NO_BLACK_PAWNS_C_FILE = NF_BLACK_PAWN * 64 + 2,
  NO_BLACK_PAWNS_D_FILE = NF_BLACK_PAWN * 64 + 3,
  NO_BLACK_PAWNS_E_FILE = NF_BLACK_PAWN * 64 + 4,
  NO_BLACK_PAWNS_F_FILE = NF_BLACK_PAWN * 64 + 5,
  NO_BLACK_PAWNS_G_FILE = NF_BLACK_PAWN * 64 + 6,
  NO_BLACK_PAWNS_H_FILE = NF_BLACK_PAWN * 64 + 7,
};

inline std::string nnue_feature_to_string(NnueFeatureBitmapType feature) {
  switch (feature) {
    case NF_WHITE_PAWN: return "White Pawn";
    case NF_WHITE_KNIGHT: return "White Knight";
    case NF_WHITE_BISHOP: return "White Bishop";
    case NF_WHITE_ROOK: return "White Rook";
    case NF_WHITE_QUEEN: return "White Queen";
    case NF_WHITE_KING: return "White King";
    case NF_WHITE_HANGING_PAWNS: return "White Hanging Pawns";
    case NF_WHITE_HANGING_KNIGHTS: return "White Hanging Knights";
    case NF_WHITE_HANGING_BISHOPS: return "White Hanging Bishops";
    case NF_WHITE_HANGING_ROOKS: return "White Hanging Rooks";
    case NF_WHITE_HANGING_QUEENS: return "White Hanging Queens";
    case NF_WHITE_HANGING_KINGS: return "White Hanging Kings";
    case NF_BLACK_PAWN: return "Black Pawn";
    case NF_BLACK_KNIGHT: return "Black Knight";
    case NF_BLACK_BISHOP: return "Black Bishop";
    case NF_BLACK_ROOK: return "Black Rook";
    case NF_BLACK_QUEEN: return "Black Queen";
    case NF_BLACK_KING: return "Black King";
    case NF_BLACK_HANGING_PAWNS: return "Black Hanging Pawns";
    case NF_BLACK_HANGING_KNIGHTS: return "Black Hanging Knights";
    case NF_BLACK_HANGING_BISHOPS: return "Black Hanging Bishops";
    case NF_BLACK_HANGING_ROOKS: return "Black Hanging Rooks";
    case NF_BLACK_HANGING_QUEENS: return "Black Hanging Queens";
    case NF_BLACK_HANGING_KINGS: return "Black Hanging Kings";
    default:
      std::cerr << "Invalid NnueFeatureBitmapType: " << feature << std::endl;
      return "Invalid Feature";
  }
}

}  // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUEFEATUREBITMAPTYPE_H
