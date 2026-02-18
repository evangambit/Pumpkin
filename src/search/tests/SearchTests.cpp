#include <gtest/gtest.h>

#include <memory>
#include <unordered_set>

#include "../search.h"
#include "../negamax.h"
#include "../../eval/pst/PieceSquareEvaluator.h"
#include "../../game/Position.h"
#include "../../game/movegen/movegen.h"

using namespace ChessEngine;

template<Color TURN>
std::pair<NegamaxResult<TURN>, std::vector<std::pair<Move, Evaluation>>> search(Position pos, std::unordered_set<Move> permittedMoves, int depth = 2, int multiPV = 1) {
  std::shared_ptr<TranspositionTable> tt = std::make_shared<TranspositionTable>(10'000);
  std::atomic<bool> stopFlag(false);
  Thread thread(0, pos, multiPV, permittedMoves, tt.get());

  std::array<Frame, 20> frames;
  NegamaxResult<TURN> result = negamax<TURN, SearchType::ROOT>(
    &thread, 2,
    ColoredEvaluation<TURN>(kMinEval),
    ColoredEvaluation<TURN>(kMaxEval),
    /*plyFromRoot=*/0,
    &frames[2],
    &stopFlag
  );
  return std::make_pair(result, thread.primaryVariations_);
}

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

// Test that NegamaxResult default constructor initializes correctly
TEST_F(SearchTest, NegamaxResultDefaultConstructor) {
  NegamaxResult<Color::WHITE> result;
  EXPECT_EQ(result.bestMove, kNullMove);
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test that NegamaxResult parameterized constructor works
TEST_F(SearchTest, NegamaxResultParameterizedConstructor) {
  Move move;
  move.from = SafeSquare::SE2;
  move.to = SafeSquare::SE4;
  move.moveType = MoveType::NORMAL;
  move.promotion = 0;
  
  NegamaxResult<Color::WHITE> result(move, ColoredEvaluation<Color::WHITE>(100));
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
  
  std::shared_ptr<TranspositionTable> tt = std::make_shared<TranspositionTable>(10'000);
  Thread thread(1, pos, 1, permittedMoves, tt.get());
  
  EXPECT_EQ(thread.id_, 1);
  EXPECT_EQ(thread.nodeCount_, 0);
  EXPECT_FALSE(thread.stopSearchFlag.load());
}

TEST_F(SearchTest, NegamaxFindsBackRankMateIn1) {
  Position pos("6k1/1RK3p1/1P3pPp/5P1P/8/8/8/8 w - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, permittedMoves).first;
  
  // The best move should be Rd4d8#.
  EXPECT_EQ(int(result.bestMove.from), int(SafeSquare::SB7));
  EXPECT_EQ(int(result.bestMove.to), int(SafeSquare::SB8));
  // Evaluation should indicate checkmate
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(-kCheckmate - 1));  // Mate in 1 ply.
}

// Test that negamax correctly identifies stalemate
// Position: Black king on a8, white king on c7, white queen on b6 - Black to move is stalemate
TEST_F(SearchTest, NegamaxIdentifiesStalemate) {
  Position pos("k7/2K5/1Q6/8/8/8/8/8 b - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  std::unordered_set<Move> permittedMoves;
  NegamaxResult<Color::BLACK> result = search<Color::BLACK>(pos, permittedMoves).first;
  
  // Stalemate should return evaluation close to 0 (draw)
  EXPECT_EQ(result.bestMove, kNullMove);
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::BLACK>(0));
}

// Test material evaluation in starting position
TEST_F(SearchTest, MaterialEvaluationStartingPosition) {
  Position pos = Position::init();
  auto evaluator = std::make_shared<SimpleEvaluator>();
  pos.set_listener(evaluator);
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}).first;
  
  // Starting position should be equal (evaluation near 0)
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test that search prefers capturing material
TEST_F(SearchTest, SearchPrefersCapturingMaterial) {
  // Position where white's bishop on f3 can capture black's queen on d5.
  Position pos("1k6/8/8/3q4/8/5B2/5K2/8 w - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  pos.set_listener(evaluator);
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}).first;
  
  // Best move should be bishop captures queen (Bxd5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SF3);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD5);
}

// Test checkmate in 3 ply.
TEST_F(SearchTest, CheckmateDetectionDepth1) {
  Position pos("r2r3k/6pp/8/8/3R4/8/8/3RK3 w - - 0 1");
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}).first;
  
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(-kCheckmate - 3));  // Mate in 3 ply.
  EXPECT_EQ(result.bestMove.from, SafeSquare::SD4);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD8);
}

// TEST_F(SearchTest, CheckmateIn3Moves) {
//   Position pos("8/7R/2r5/8/P3n3/8/3nk1PP/1R4K1 b - - 1 1");
//   pos.set_listener(std::make_shared<SimpleEvaluator>());
//   NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}, 4).first;
//   EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(-kCheckmate - 5));  // Mate in 5 ply.
//   EXPECT_EQ(int(result.bestMove.from), int(SafeSquare::SD2));
//   EXPECT_EQ(int(result.bestMove.to), int(SafeSquare::SF3));
// }

// Test fifty move rule detection
TEST_F(SearchTest, FiftyMoveRuleDetection) {
  // We give white a bishop and a pawn to drawing by insufficient material.
  Position pos("k7/P7/8/8/8/K1B5/8/8 w - - 98 120");
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  // With depth > 0, should detect fifty move rule and return draw
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}, 2).first;
  
  // Position should be evaluated as draw due to fifty move rule
  // (after any move, fifty move rule kicks in)
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test negamax with black to move
TEST_F(SearchTest, NegamaxBlackToMove) {
  // Position where black can capture white's queen with a fork.
  Position pos("4kr2/4n3/3Q4/8/8/4K3/8/8 b - - 0 1");
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  NegamaxResult<Color::BLACK> result = search<Color::BLACK>(pos, {}).first;
  // Best move should be knight forking king and queen (Ne7-f5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SE7);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SF5);

  make_move<Color::BLACK>(&pos, result.bestMove);
  make_move<Color::WHITE>(&pos, make_move_from_uci("e3e4", pos));

  result = search<Color::BLACK>(pos, {}).first;
  // Next best move should be knight captures queen (Nxf5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SF5);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD6);
}

// Test that search handles check correctly
TEST_F(SearchTest, SearchHandlesCheck) {
  // Black is in check. He has lots of pieces but only one legal move to get out of check (Kf7).
  Position pos("3bkbr1/4b3/8/1B6/5q2/8/K7/8 b - - 0 1");
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  NegamaxResult<Color::BLACK> result = search<Color::BLACK>(pos, {}).first;
  
  EXPECT_EQ(result.bestMove.from, SafeSquare::SE8);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SF7);
}

TEST_F(SearchTest, PermittedMovesFilterRestrictsSearch) {
  Position pos = Position::init();
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  
  // Only permit the move c2c3.
  Move permittedMove;
  permittedMove.from = SafeSquare::SC2;
  permittedMove.to = SafeSquare::SC3;
  permittedMove.moveType = MoveType::NORMAL;
  permittedMove.promotion = 0;

  std::unordered_set<Move> permittedMoves;
  permittedMoves.insert(permittedMove);
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, permittedMoves).first;
  
  // Best move should be forced to c2c3 since that's the only permitted move.
  EXPECT_EQ(result.bestMove, permittedMove);
}

// Test that empty permitted moves set allows all moves
TEST_F(SearchTest, EmptyPermittedMovesAllowsAllMoves) {
  // Same position as above - without restrictions, Bf3xd5 should be chosen
  Position pos("3k4/8/8/3q4/8/5B2/8/1K6 w - - 0 1");
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}).first;
  
  // Best move should be bishop captures queen (Bf3xd5)
  EXPECT_EQ(result.bestMove.from, SafeSquare::SF3);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD5);
}

// Test permitted moves with multiple allowed moves
TEST_F(SearchTest, PermittedMovesWithMultipleMoves) {
  // Position where white can capture queen with Bf3xd5 or make various king moves
  Position pos("3k4/8/8/3q4/8/5B2/8/1K6 w - - 0 1");
  pos.set_listener(std::make_shared<SimpleEvaluator>());
  
  // Permit both the bishop capture and a king move
  Move bishopCapture;
  bishopCapture.from = SafeSquare::SF3;
  bishopCapture.to = SafeSquare::SD5;
  bishopCapture.moveType = MoveType::NORMAL;
  bishopCapture.promotion = 0;
  
  Move kingMove;
  kingMove.from = SafeSquare::SF2;
  kingMove.to = SafeSquare::SE3;
  kingMove.moveType = MoveType::NORMAL;
  kingMove.promotion = 0;
  
  std::unordered_set<Move> permittedMoves;
  permittedMoves.insert(bishopCapture);
  permittedMoves.insert(kingMove);
  
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, permittedMoves).first;
  
  // Best move should still be Bxd5 since it captures the queen
  EXPECT_EQ(result.bestMove.from, SafeSquare::SF3);
  EXPECT_EQ(result.bestMove.to, SafeSquare::SD5);
}

// Test that PieceSquareEvaluator evaluates the starting position to 0
// Searches to a depth of 4, then checks that the evaluation is 0
TEST_F(SearchTest, PieceSquareEvaluatorStartingPosition) {
  Position pos = Position::init();
  pos.set_listener(std::make_shared<PieceSquareEvaluator>());
  NegamaxResult<Color::WHITE> result = search<Color::WHITE>(pos, {}).first;
  
  // Starting position should be evaluated as equal (0)
  EXPECT_EQ(result.evaluation, ColoredEvaluation<Color::WHITE>(0));
}

// Test that transposition table is used and stores entries
TEST_F(SearchTest, TranspositionTableStoresAndProbes) {
  Position pos = Position::init();
  auto evaluator = std::make_shared<SimpleEvaluator>();
  TranspositionTable tt(1024); // Small table for test
  // First search: should fill the table
  auto result1 = search(pos, evaluator, 2, 1, &tt);
  // Probe directly
  TTEntry entry;
  bool found = tt.probe(pos.currentState_.hash, entry);
  EXPECT_TRUE(found);
  EXPECT_EQ(entry.depth, 2);
  EXPECT_EQ(entry.bestMove, result1.bestMove);
  EXPECT_EQ(entry.value, result1.evaluation.value);
  // Second search: should hit the table
  auto result2 = search(pos, evaluator, 2, 1, &tt);
  // Results should be the same
  EXPECT_EQ(result1.bestMove, result2.bestMove);
  EXPECT_EQ(result1.evaluation, result2.evaluation);
}


// MultiPV tests
namespace {

class MultiPVTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

TEST_F(MultiPVTest, MultiPVReturnsTopNMoves) {
  // White can capture black queen with c4d5, e4d5, or e3d5
  Position pos("2k5/8/8/3q4/2P1P3/4N3/8/K7 w - - 0 1");
  auto evaluator = std::make_shared<SimpleEvaluator>();
  int multiPV = 3;
  std::shared_ptr<TranspositionTable> tt = std::make_shared<TranspositionTable>(10'000);
  pos.set_listener(evaluator);
  auto result = search<Color::WHITE>(pos, {}, 1, multiPV);
  
  // Should have exactly multiPV moves in primaryVariations_
  EXPECT_EQ(result.second.size(), multiPV);
  ASSERT_GE(result.second[0].second, result.second[1].second);
  ASSERT_GE(result.second[1].second, result.second[2].second);

  std::unordered_set<std::string> expectedMoves = {"c4d5", "e4d5", "e3d5"};
  std::unordered_set<std::string> foundMoves;
  for (const auto& entry : result.second) {
    foundMoves.insert(entry.first.uci());
  }
  EXPECT_EQ(foundMoves, expectedMoves);
}

} // anonymous namespace
