#include <gtest/gtest.h>

#include <memory>
#include <unordered_set>

#include "../search.h"
#include "../../game/Position.h"
#include "../../game/movegen/movegen.h"

namespace ChessEngine {

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

// Helper to create a move from UCI string
Move make_move_from_uci(const std::string& uci, const Position& pos) {
  Move move;
  move.from = SafeSquare((7 - (uci[1] - '1')) * 8 + (uci[0] - 'a'));
  move.to = SafeSquare((7 - (uci[3] - '1')) * 8 + (uci[2] - 'a'));
  move.moveType = MoveType::NORMAL;
  move.promotion = 0;
  
  // Check for promotion
  if (uci.length() == 5) {
    move.moveType = MoveType::PROMOTION;
    switch (uci[4]) {
      case 'n': move.promotion = 0; break;
      case 'b': move.promotion = 1; break;
      case 'r': move.promotion = 2; break;
      case 'q': move.promotion = 3; break;
    }
  }
  
  // Check for castling
  ColoredPiece cp = pos.tiles_[move.from];
  if (cp == ColoredPiece::WHITE_KING || cp == ColoredPiece::BLACK_KING) {
    if (abs(int(move.from % 8) - int(move.to % 8)) == 2) {
      move.moveType = MoveType::CASTLE;
    }
  }
  
  // Check for en passant
  if ((cp == ColoredPiece::WHITE_PAWN || cp == ColoredPiece::BLACK_PAWN) &&
      (move.from % 8) != (move.to % 8) &&
      pos.tiles_[move.to] == ColoredPiece::NO_COLORED_PIECE) {
    move.moveType = MoveType::EN_PASSANT;
  }
  
  return move;
}

// Test that SearchResult default constructor initializes correctly
TEST_F(SearchTest, SearchResultDefaultConstructor) {
  SearchResult<Color::WHITE> result;
  EXPECT_EQ(result.bestMove, kNullMove);
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test that SearchResult parameterized constructor works
TEST_F(SearchTest, SearchResultParameterizedConstructor) {
  Move move;
  move.from = SafeSquare::SE2;
  move.to = SafeSquare::SE4;
  move.moveType = MoveType::NORMAL;
  move.promotion = 0;
  
  SearchResult<Color::WHITE> result(move, 100);
  EXPECT_EQ(result.bestMove.from, SafeSquare::SE2);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SE4);
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(100));
}

// Test SimpleEvaluator piece values are initialized
TEST_F(SearchTest, SimpleEvaluatorPieceValues) {
  // White pieces should have positive values
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::WHITE_PAWN], ColoredEvaluation<Color::WHITE>(100));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::WHITE_KNIGHT], ColoredEvaluation<Color::WHITE>(320));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::WHITE_BISHOP], ColoredEvaluation<Color::WHITE>(330));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::WHITE_ROOK], ColoredEvaluation<Color::WHITE>(500));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::WHITE_QUEEN], ColoredEvaluation<Color::WHITE>(900));
  
  // Black pieces should have negative values
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::BLACK_PAWN], ColoredEvaluation<Color::WHITE>(-100));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::BLACK_KNIGHT], ColoredEvaluation<Color::WHITE>(-320));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::BLACK_BISHOP], ColoredEvaluation<Color::WHITE>(-330));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::BLACK_ROOK], ColoredEvaluation<Color::WHITE>(-500));
  EXPECT_EQ(SimpleEvaluator::kPieceValues[ColoredPiece::BLACK_QUEEN], ColoredEvaluation<Color::WHITE>(-900));
}

// Test Thread constructor
TEST_F(SearchTest, ThreadConstructor) {
  Position pos = Position::init();
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(1, pos, evaluator, permittedMoves);
  
  EXPECT_EQ(thread.id_, 1);
  EXPECT_EQ(thread.nodeCount_, 0);
  EXPECT_FALSE(thread.stopSearchFlag.load());
}

TEST_F(SearchTest, NegamaxFindsBackRankMateIn1) {
  Position pos("6k1/5ppp/8/8/3Q4/8/8/4K3 w - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  SearchResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(&thread, 2, ColoredEvaluation<Color::WHITE>(kMinEval), ColoredEvaluation<Color::WHITE>(kMaxEval));
  
  // The best move should be Qd4d8#.
  EXPECT_EQ(result.bestMove.from, SafeSquare::SD4);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD8);
  // Evaluation should indicate checkmate
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(-kCheckmate - 1));  // Mate in 1 ply.
}

// Test that negamax correctly identifies stalemate
// Position: Black king on a8, white king on c7, white queen on b6 - Black to move is stalemate
TEST_F(SearchTest, NegamaxIdentifiesStalemate) {
  Position pos("k7/2K5/1Q6/8/8/8/8/8 b - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  SearchResult<Color::BLACK> result = negamax<Color::BLACK, SearchType::ROOT>(&thread, 1, ColoredEvaluation<Color::BLACK>(kMinEval), ColoredEvaluation<Color::BLACK>(kMaxEval));
  
  // Stalemate should return evaluation close to 0 (draw)
  EXPECT_EQ(result.bestMove, kNullMove);
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::BLACK>(0));
}

// Test material evaluation in starting position
TEST_F(SearchTest, MaterialEvaluationStartingPosition) {
  Position pos = Position::init();
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  // At depth 0, should just evaluate the position
  SearchResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(&thread, 0, ColoredEvaluation<Color::WHITE>(kMinEval), ColoredEvaluation<Color::WHITE>(kMaxEval));
  
  // Starting position should be equal (evaluation near 0)
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test that search prefers capturing material
TEST_F(SearchTest, SearchPrefersCapturingMaterial) {
  // Position where white's bishop on f3 can capture black's queen on d5.
  Position pos("1k6/8/8/3q4/8/5B2/5K2/8 w - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  SearchResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(&thread, 2, ColoredEvaluation<Color::WHITE>(kMinEval), ColoredEvaluation<Color::WHITE>(kMaxEval));
  
  // Best move should be bishop captures queen (Bxd5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SF3);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD5);
}

// Test checkmate in 3 ply.
TEST_F(SearchTest, CheckmateDetectionDepth1) {
  Position pos("r2r3k/6pp/8/8/3R4/8/8/3RK3 w - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  SearchResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(&thread, 4, ColoredEvaluation<Color::WHITE>(kMinEval), ColoredEvaluation<Color::WHITE>(kMaxEval));
  
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(-kCheckmate - 3));  // Mate in 3 ply.
  EXPECT_EQ(result.bestMove.from, SafeSquare::SD4);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD8);
}

// Test fifty move rule detection
TEST_F(SearchTest, FiftyMoveRuleDetection) {
  // We give white a bishop and a pawn to drawing by insufficient material.
  Position pos("k7/P7/8/8/8/K1B5/8/8 w - - 98 120");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  // With depth > 0, should detect fifty move rule and return draw
  SearchResult<Color::WHITE> result = negamax<Color::WHITE, SearchType::ROOT>(&thread, 3, ColoredEvaluation<Color::WHITE>(kMinEval), ColoredEvaluation<Color::WHITE>(kMaxEval));
  
  // Position should be evaluated as draw due to fifty move rule
  // (after any move, fifty move rule kicks in)
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test negamax with black to move
TEST_F(SearchTest, NegamaxBlackToMove) {
  // Position where black can capture white's queen with a fork.
  Position pos("4kr2/4n3/3Q4/8/8/4K3/8/8 b - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  SearchResult<Color::BLACK> result = negamax<Color::BLACK, SearchType::ROOT>(&thread, 2, ColoredEvaluation<Color::BLACK>(kMinEval), ColoredEvaluation<Color::BLACK>(kMaxEval));
  
  // Best move should be knight forking king and queen (Ne7-f5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SE7);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SF5);

  make_move<Color::BLACK>(&thread.position_, result.bestMove);
  make_move<Color::WHITE>(&thread.position_, make_move_from_uci("e3e4", thread.position_));

  result = negamax<Color::BLACK, SearchType::ROOT>(&thread, 2, ColoredEvaluation<Color::BLACK>(kMinEval), ColoredEvaluation<Color::BLACK>(kMaxEval));
  // Next best move should be knight captures queen (Nxf5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SF5);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD6);
}

// Test that search handles check correctly
TEST_F(SearchTest, SearchHandlesCheck) {
  // Black is in check. He has lots of pieces but only one legal move to get out of check (Kf7).
  Position pos("3bkbr1/4b3/8/1B6/5q2/8/K7/8 b - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  
  Thread thread(0, pos, evaluator, permittedMoves);
  
  SearchResult<Color::BLACK> result = negamax<Color::BLACK, SearchType::ROOT>(&thread, 1, ColoredEvaluation<Color::BLACK>(kMinEval), ColoredEvaluation<Color::BLACK>(kMaxEval));
  
  EXPECT_EQ(result.bestMove.from, SafeSquare::SE8);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SF7);
}

}  // namespace ChessEngine
