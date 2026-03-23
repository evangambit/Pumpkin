#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>

#include "../byhand.h"
#include "../../../game/Position.h"
#include "../../../game/CreateThreats.h"
#include "../../../game/movegen/movegen.h"

using namespace ChessEngine::ByHand;
using namespace ChessEngine;

class ByHandTests : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

std::vector<int16_t> get_feature_vector(std::string fen) {
  Position pos(fen);
  Threats threats;
  create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
  int8_t out[EF::EF_COUNT];
  if (pos.turn_ == Color::WHITE) {
    pos2features<Color::WHITE>(pos, threats, out);
  } else {
    pos2features<Color::BLACK>(pos, threats, out);
  }
  std::vector<int16_t> r;
  for (size_t i = 0; i < EF::EF_COUNT; ++i) {
    r.push_back(out[i]);
  }
  return r;
}

// TEST_F(ByHandTests, TestBackwardPawns) {
//   EXPECT_EQ(get_feature_vector("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")[EF::BACKWARD_PAWN], 0);
//   EXPECT_EQ(get_feature_vector("5k2/3p4/2p5/P7/1P6/8/8/4K3 w - - 0 1")[EF::BACKWARD_PAWN], 1);
// }

// TEST_F(ByHandTests, TestStragglerPawns) {
//   EXPECT_EQ(get_feature_vector("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")[EF::STRAGGLER_PAWN], 0);
//   EXPECT_EQ(get_feature_vector("5k2/3p4/2p5/P7/1P6/8/8/4K3 w - - 0 1")[EF::STRAGGLER_PAWN], 1);
// }

// TEST_F(ByHandTests, TestCandidatePassedPawns) {
//   EXPECT_EQ(get_feature_vector("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")[EF::CANDIDATE_PASSED_PAWN], 0);
//   EXPECT_EQ(get_feature_vector("4k3/8/6p1/p1p5/7P/PPP3P1/8/4K3 w - - 0 1")[EF::CANDIDATE_PASSED_PAWN], 2);
// }

TEST_F(ByHandTests, TestOutposts) {
  EXPECT_EQ(get_feature_vector("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")[EF::FIFTH_RANK_OUTPOST], 0);
  EXPECT_EQ(get_feature_vector("1k6/8/3n4/3p4/3P1P2/4P3/8/1K6 w - - 0 1")[EF::FIFTH_RANK_OUTPOST], -2);
  // Knight cannot get to outpost squares.
  EXPECT_EQ(get_feature_vector("1k6/8/8/3p1n2/3P1P2/4P3/8/1K6 w - - 0 1")[EF::FIFTH_RANK_OUTPOST], 0);
}
