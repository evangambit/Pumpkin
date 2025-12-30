#include <gtest/gtest.h>

#include <vector>

#include "../Position.h"
#include "../utils.h"

namespace ChessEngine {

struct FakeBoardListener : public BoardListener {
  std::vector<std::pair<ColoredPiece, SafeSquare>> placedPieces;
  std::vector<SafeSquare> removedSquares;
  bool wasEmptied = false;
  void empty() override {
    wasEmptied = true;
  }
  void place_piece(ColoredPiece cp, SafeSquare square) override {
    placedPieces.push_back({cp, square});
  }
  void remove_piece(SafeSquare square) override {
    removedSquares.push_back(square);
  }
};

class PositionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
  }
};

// Test default constructor creates empty position with white to move
TEST_F(PositionTest, DefaultConstructor) {
  Position pos;
  EXPECT_EQ(pos.turn_, Color::WHITE);
  EXPECT_EQ(pos.currentState_.hash, 0);
  for (size_t i = 0; i < kNumSquares; ++i) {
    EXPECT_EQ(pos.tiles_[i], ColoredPiece::NO_COLORED_PIECE);
  }
  EXPECT_EQ(pos.wholeMoveCounter_, 1);
  EXPECT_EQ(pos.currentState_.halfMoveCounter, 0);
  EXPECT_EQ(pos.currentState_.epSquare, UnsafeSquare::UNO_SQUARE);
  EXPECT_EQ(pos.currentState_.castlingRights, kCastlingRights_NoRights);
}

// Test Position::init() creates standard starting position
TEST_F(PositionTest, InitCreatesStartingPosition) {
  Position pos = Position::init();
  
  EXPECT_EQ(pos.turn_, Color::WHITE);
  EXPECT_EQ(pos.wholeMoveCounter_, 1);
  EXPECT_EQ(pos.currentState_.halfMoveCounter, 0);
  EXPECT_EQ(pos.currentState_.epSquare, UnsafeSquare::UNO_SQUARE);
  
  // Check castling rights - all should be available at start
  EXPECT_TRUE(pos.currentState_.castlingRights & kCastlingRights_WhiteKing);
  EXPECT_TRUE(pos.currentState_.castlingRights & kCastlingRights_WhiteQueen);
  EXPECT_TRUE(pos.currentState_.castlingRights & kCastlingRights_BlackKing);
  EXPECT_TRUE(pos.currentState_.castlingRights & kCastlingRights_BlackQueen);
}

// Test FEN parsing for starting position
TEST_F(PositionTest, FenParsingStartingPosition) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  EXPECT_EQ(pos.turn_, Color::WHITE);
  EXPECT_EQ(pos.wholeMoveCounter_, 1);
  
  // Verify piece placement
  EXPECT_EQ(pos.tiles_[SafeSquare::SA8], ColoredPiece::BLACK_ROOK);
  EXPECT_EQ(pos.tiles_[SafeSquare::SE8], ColoredPiece::BLACK_KING);
  EXPECT_EQ(pos.tiles_[SafeSquare::SA1], ColoredPiece::WHITE_ROOK);
  EXPECT_EQ(pos.tiles_[SafeSquare::SE1], ColoredPiece::WHITE_KING);
}

// Test BoardListener is given all pieces on initialization
TEST_F(PositionTest, BoardListenerInitialization) {
  Position pos = Position::init();
  auto listener = std::make_shared<FakeBoardListener>();
  pos.set_listener(listener);
  
  EXPECT_TRUE(listener->wasEmptied);
  EXPECT_EQ(listener->placedPieces.size(), 32);  // 32 pieces in starting position
}

// Test FEN parsing with black to move
TEST_F(PositionTest, FenParsingBlackToMove) {
  Position pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  
  EXPECT_EQ(pos.turn_, Color::BLACK);
  EXPECT_EQ(pos.currentState_.epSquare, SafeSquare::SE3);
}

// Test FEN generation matches input
TEST_F(PositionTest, FenRoundTrip) {
  std::string originalFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  Position pos(originalFen);
  
  EXPECT_EQ(pos.fen(), originalFen);
}

Move e2e4() {
  Move move;
  move.from = SafeSquare::SE2;  // e2 in 0-indexed from top
  move.to = SafeSquare::SE4;    // e4
  move.moveType = MoveType::NORMAL;
  move.promotion = 0;
  return move;
}

// Test make_move and undo for a simple pawn move
TEST_F(PositionTest, MakeMoveAndUndo) {
  Position pos = Position::init();
  Move move = e2e4();
  
  uint64_t originalHash = pos.currentState_.hash;
  const uint64_t expectedHashAfterMove = originalHash 
    ^ kZorbristNumbers[ColoredPiece::WHITE_PAWN][SafeSquare::SE2] 
    ^ kZorbristNumbers[ColoredPiece::WHITE_PAWN][SafeSquare::SE4]
    ^ kZorbristEnpassant[SafeSquare::SE3 % 8] 
    ^ kZorbristTurn;
  
  make_move<Color::WHITE>(&pos, move);
  
  EXPECT_EQ(pos.turn_, Color::BLACK);
  EXPECT_EQ(pos.tiles_[SafeSquare::SE2], ColoredPiece::NO_COLORED_PIECE);
  EXPECT_EQ(pos.tiles_[SafeSquare::SE4], ColoredPiece::WHITE_PAWN);
  if (pos.currentState_.hash != expectedHashAfterMove) {
    print_zorbrist_debug(pos.currentState_.hash, expectedHashAfterMove);
    EXPECT_TRUE(false);
    return;
  }
  
  undo<Color::WHITE>(&pos);
  
  EXPECT_EQ(pos.turn_, Color::WHITE);
  EXPECT_EQ(pos.tiles_[SafeSquare::SE2], ColoredPiece::WHITE_PAWN);
  EXPECT_EQ(pos.tiles_[SafeSquare::SE4], ColoredPiece::NO_COLORED_PIECE);
  if (pos.currentState_.hash != originalHash) {
    print_zorbrist_debug(pos.currentState_.hash, originalHash);
    EXPECT_TRUE(false);
    return;
  }
}

// Test null move
TEST_F(PositionTest, NullMove) {
  Position pos = Position::init();
  
  make_nullmove<Color::WHITE>(&pos);
  
  EXPECT_EQ(pos.turn_, Color::BLACK);
  EXPECT_EQ(pos.currentState_.epSquare, UnsafeSquare::UNO_SQUARE);
  
  undo_nullmove<Color::WHITE>(&pos);
  
  EXPECT_EQ(pos.turn_, Color::WHITE);
}

// Test null move after a pawn double move
TEST_F(PositionTest, NullMoveAfterPawnDouble) {
  Position pos = Position::init();
  
  Move move = e2e4();
  make_move<Color::WHITE>(&pos, move);

  uint64_t hashAfterWhiteMove = pos.currentState_.hash;
  EXPECT_EQ(pos.turn_, Color::BLACK);
  EXPECT_EQ(pos.currentState_.epSquare, SafeSquare::SE3);

  make_nullmove<Color::BLACK>(&pos);

  uint64_t hashAfterBlackNullMove = pos.currentState_.hash;
  EXPECT_EQ(hashAfterBlackNullMove, hashAfterWhiteMove ^ kZorbristTurn ^ kZorbristEnpassant[SafeSquare::SE3 % 8]);
  EXPECT_EQ(pos.turn_, Color::WHITE);
  EXPECT_EQ(pos.currentState_.epSquare, UnsafeSquare::UNO_SQUARE);
  
  undo_nullmove<Color::BLACK>(&pos);

  EXPECT_EQ(pos.currentState_.hash, hashAfterWhiteMove);
  EXPECT_EQ(pos.turn_, Color::BLACK);
  EXPECT_EQ(pos.currentState_.epSquare, SafeSquare::SE3);

  undo<Color::WHITE>(&pos);

  EXPECT_EQ(pos.currentState_.hash, Position::init().currentState_.hash);
  EXPECT_EQ(pos.turn_, Color::WHITE);
  EXPECT_EQ(pos.currentState_.epSquare, UnsafeSquare::UNO_SQUARE);
}

// Test Move UCI string generation
TEST_F(PositionTest, MoveUciString) {
  Move move;
  move.from = SafeSquare::SE2;
  move.to = SafeSquare::SE4;
  move.moveType = MoveType::NORMAL;
  move.promotion = 0;
  
  // Note: The actual string depends on square_to_string implementation
  EXPECT_FALSE(move.uci().empty());
}

// Test material draw detection - King vs King
TEST_F(PositionTest, MaterialDrawKingVsKing) {
  Position pos("8/8/8/4k3/8/8/8/4K3 w - - 0 1");
  
  EXPECT_TRUE(pos.is_material_draw());
}

// Test material draw detection - King + Bishop vs King
TEST_F(PositionTest, MaterialDrawKingBishopVsKing) {
  Position pos("8/8/8/4k3/8/8/2B5/4K3 w - - 0 1");
  
  EXPECT_TRUE(pos.is_material_draw());
}

// Test material draw detection - King + Knight vs King
TEST_F(PositionTest, MaterialDrawKingKnightVsKing) {
  Position pos("8/8/8/4k3/8/8/2N5/4K3 w - - 0 1");
  
  EXPECT_TRUE(pos.is_material_draw());
}

// Test non-draw position
TEST_F(PositionTest, NotMaterialDraw) {
  Position pos = Position::init();
  
  EXPECT_FALSE(pos.is_material_draw());
}

// Test fifty move rule
TEST_F(PositionTest, FiftyMoveRule) {
  Position pos("8/8/8/4k3/8/8/4Q3/4K3 w - - 100 50");
  
  EXPECT_TRUE(pos.is_fifty_move_rule());
}

// Test not fifty move rule
TEST_F(PositionTest, NotFiftyMoveRule) {
  Position pos("8/8/8/4k3/8/8/4Q3/4K3 w - - 49 25");
  
  EXPECT_FALSE(pos.is_fifty_move_rule());
}

}  // namespace ChessEngine
