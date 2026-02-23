#include <gtest/gtest.h>

#include <vector>

#include "../Position.h"
#include "../Utils.h"
#include "../Threats.h"
#include "../CreateThreats.h"

#define EXPECT_BB_EQ(actual, expected) \
  EXPECT_TRUE((actual) == (expected)) \
    << "Bitboards differ:\n" \
    << "Actual:\n" << bstr(actual) \
    << "Expected:\n" << bstr(expected)

namespace ChessEngine {

class ThreatsTests : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
  }
};

// Test not fifty move rule
TEST_F(ThreatsTests, NotFiftyMoveRule) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  Threats threats;
  create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
  EXPECT_BB_EQ(threats.badForOur<Color::WHITE>(Piece::PAWN), kRanks[2]);
}

}  // namespace ChessEngine
