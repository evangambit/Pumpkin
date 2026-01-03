#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <string>

#include "../../Position.h"
#include "../movegen.h"

namespace ChessEngine {

class MoveGenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_movegen();
    initialize_geometry();
  }

  // Helper to check if a specific move exists in the move list
  bool containsMove(ExtMove* begin, ExtMove* end, SafeSquare from, SafeSquare to) {
    for (ExtMove* m = begin; m != end; ++m) {
      if (m->move.from == from && m->move.to == to) {
        return true;
      }
    }
    return false;
  }

  // Helper to check if a promotion move exists
  bool containsPromotion(ExtMove* begin, ExtMove* end, SafeSquare from, SafeSquare to, Piece promoPiece) {
    for (ExtMove* m = begin; m != end; ++m) {
      if (m->move.from == from && m->move.to == to && 
          m->move.moveType == MoveType::PROMOTION &&
          m->move.promotion == (promoPiece - 2)) {  // promotion encodes knight=0, bishop=1, rook=2, queen=3
        return true;
      }
    }
    return false;
  }

  // Helper to count moves
  int countMoves(ExtMove* begin, ExtMove* end) {
    return static_cast<int>(end - begin);
  }
};

// // Test 1: Starting position should have 20 legal moves for white
// TEST_F(MoveGenTest, StartingPositionHas20Moves) {
//   Position pos = Position::init();
//   ExtMove moves[kMaxNumMoves];
  
//   ExtMove* end = compute_legal_moves<Color::WHITE>(&pos, moves);
//   int numMoves = countMoves(moves, end);
  
//   EXPECT_EQ(numMoves, 20);
  
//   // Check that all pawn moves exist (8 single pushes + 8 double pushes = 16 pawn moves)
//   // Plus 4 knight moves (2 knights x 2 moves each)
  
//   // Verify some specific pawn moves
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SE2, SafeSquare::SE3));  // e2-e3
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SE2, SafeSquare::SE4));  // e2-e4
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SD2, SafeSquare::SD4));  // d2-d4
  
//   // Verify knight moves
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SG1, SafeSquare::SF3));  // Nf3
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SG1, SafeSquare::SH3));  // Nh3
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SB1, SafeSquare::SC3));  // Nc3
//   EXPECT_TRUE(containsMove(moves, end, SafeSquare::SB1, SafeSquare::SA3));  // Na3
// }

// // Test 2: Pawn promotion generates all four promotion options
// TEST_F(MoveGenTest, PawnPromotionGeneratesAllOptions) {
//   // White pawn on e7, white king on e1, black king on e4 (not blocking e8)
//   Position pos("8/4P3/8/8/4k3/8/8/4K3 w - - 0 1");
//   ExtMove moves[kMaxNumMoves];
  
//   ExtMove* end = compute_legal_moves<Color::WHITE>(&pos, moves);
  
//   // Should have pawn promotions to e8 (4 options) plus king moves
//   // Check all four promotion types exist
//   EXPECT_TRUE(containsPromotion(moves, end, SafeSquare::SE7, SafeSquare::SE8, Piece::QUEEN));
//   EXPECT_TRUE(containsPromotion(moves, end, SafeSquare::SE7, SafeSquare::SE8, Piece::ROOK));
//   EXPECT_TRUE(containsPromotion(moves, end, SafeSquare::SE7, SafeSquare::SE8, Piece::BISHOP));
//   EXPECT_TRUE(containsPromotion(moves, end, SafeSquare::SE7, SafeSquare::SE8, Piece::KNIGHT));
// }

// Test 3: King in check must escape - only valid escape moves generated
TEST_F(MoveGenTest, KingInCheckMustEscape) {
  // White king on e1 with black rook giving check on e8
  // White has a knight on c3 that can block
  Position pos("2k1r3/8/8/8/8/2N5/8/4K3 w - - 0 1");
  ExtMove moves[kMaxNumMoves];
  
  ExtMove* end = compute_legal_moves<Color::WHITE>(&pos, moves);
  int numMoves = countMoves(moves, end);
  
  // King must move or knight must block on e2/e4/e5/e6/e7
  // King can move to d1, d2, f1, f2 (4 moves, e2 is blocked by check)
  // Knight can block on e2 or e4 (2 blocking moves)
  // Total: should be limited moves due to check

  std::vector<std::string> expectedMoves = {
    "e1d1", "e1f1", "e1d2", "e1f2",  // King moves
    "c3e2", "c3e4"                   // Knight blocks
  };

  std::vector<std::string> generatedMoves;
  for (ExtMove* m = moves; m != end; ++m) {
    generatedMoves.push_back(m->move.uci());
  }

  std::sort(generatedMoves.begin(), generatedMoves.end());
  std::sort(expectedMoves.begin(), expectedMoves.end());

  EXPECT_EQ(generatedMoves, expectedMoves);

}

// Enpassant into check is illegal
TEST_F(MoveGenTest, EnpassantIntoCheckIsIllegal) {
  // White pawn on e5, black pawn on d7, black rook on e8 giving check
  Position pos("r4Q2/1pk2pp1/8/3qpP1K/8/8/PP5B/n7 w - - 0 25");
  ExtMove moves[kMaxNumMoves];
  ExtMove* end = compute_legal_moves<Color::WHITE>(&pos, moves);
  // White should not be able to play e5xd6 enpassant as it would leave king in check
  EXPECT_FALSE(containsMove(moves, end, SafeSquare::SE5, SafeSquare::SD6));
}

}  // namespace ChessEngine
