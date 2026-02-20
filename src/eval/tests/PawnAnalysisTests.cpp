#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>

#include "../../game/Position.h"
#include "../../game/movegen/movegen.h"
#include "../PawnAnalysis.h"

using namespace ChessEngine;

#define EXPECT_BB_EQ(actual, expected) \
  EXPECT_TRUE((actual) == (expected)) \
    << "Bitboards differ:\n" \
    << "Actual:\n" << bstr(actual) \
    << "Expected:\n" << bstr(expected)

class PawnAnalysisTests : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

TEST_F(PawnAnalysisTests, PassedPawns) {
  {
    Position pos = Position::init();
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_EQ(analysis.ourPassedPawns, kEmptyBitboard);
    EXPECT_EQ(analysis.theirPassedPawns, kEmptyBitboard);
  }
  {
    Position pos("8/8/7p/8/4pP2/4P3/8/8 w - - 0 1");
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_BB_EQ(analysis.ourPassedPawns, bb(SafeSquare::SF4));
    EXPECT_BB_EQ(analysis.theirPassedPawns, bb(SafeSquare::SH6));
  }
}

TEST_F(PawnAnalysisTests, IsolatedPawns) {
  {
    Position pos = Position::init();
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_EQ(analysis.ourIsolatedPawns, kEmptyBitboard);
    EXPECT_EQ(analysis.theirIsolatedPawns, kEmptyBitboard);
  }
  {
    Position pos("8/8/8/2p1p3/8/8/8/8 w - - 0 1");
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_BB_EQ(analysis.ourIsolatedPawns, kEmptyBitboard);
    EXPECT_BB_EQ(analysis.theirIsolatedPawns, bb(SafeSquare::SC5) | bb(SafeSquare::SE5));
  }
}

TEST_F(PawnAnalysisTests, DoubledPawns) {
  {
    Position pos = Position::init();
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_EQ(analysis.ourDoubledPawns, kEmptyBitboard);
    EXPECT_EQ(analysis.theirDoubledPawns, kEmptyBitboard);
  }
  {
    Position pos("8/8/8/2p1p3/4p3/8/8/8 w - - 0 1");
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_BB_EQ(analysis.ourDoubledPawns, kEmptyBitboard);
    EXPECT_BB_EQ(analysis.theirDoubledPawns, bb(SafeSquare::SE4));
  }
}

TEST_F(PawnAnalysisTests, Outposts) {
  {
    Position pos = Position::init();
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_BB_EQ(analysis.ourOutposts, kEmptyBitboard);
    EXPECT_BB_EQ(analysis.theirOutposts, kEmptyBitboard);
  }
  {
    Position pos("rnbqkbnr/pp1p1ppp/8/2p1p3/2P1P3/8/PP1P1PPP/RNBQKBNR w KQkq - 0 1");
    PawnAnalysis<Color::WHITE> analysis(pos);
    EXPECT_BB_EQ(analysis.ourOutposts, bb(SafeSquare::SD5));
    EXPECT_BB_EQ(analysis.theirOutposts, bb(SafeSquare::SD4));
  }
}
