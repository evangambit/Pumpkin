#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#include "../QstEvaluator.h"
#include "../../../game/Position.h"
#include "../../../game/movegen/movegen.h"
#include "../../../game/Threats.h"

using namespace ChessEngine;

#define EXPECT_BB_EQ(actual, expected) \
  EXPECT_TRUE((actual) == (expected)) \
    << "Bitboards differ:\n" \
    << "Actual:\n" << bstr(actual) \
    << "Expected:\n" << bstr(expected)

class QstEvaluatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

TEST_F(QstEvaluatorTest, FeatureCountMatchesExpected) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  QstEvaluator qstEvaluator;
  std::vector<Bitboard> features;
  qstEvaluator.get_features<Color::WHITE>(pos, &features);
  EXPECT_EQ(features.size(), 44);
}

constexpr Bitboard WHITE_PAWNS_AFTER_E4 = bb(SA2) | bb(SB2) | bb(SC2) | bb(SD2) | bb(SE4) | bb(SF2) | bb(SG2) | bb(SH2);
constexpr Bitboard BLACK_PAWNS_START = bb(SA7) | bb(SB7) | bb(SC7) | bb(SD7) | bb(SE7) | bb(SF7) | bb(SG7) | bb(SH7);

TEST_F(QstEvaluatorTest, PieceFeatures) {
  // 1. e4 Nc6
  Position pos("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2");
  QstEvaluator qstEvaluator;
  std::vector<Bitboard> features;
  qstEvaluator.get_features<Color::WHITE>(pos, &features);
  EXPECT_BB_EQ(features[0], WHITE_PAWNS_AFTER_E4);
  EXPECT_BB_EQ(features[1], BLACK_PAWNS_START);
}

TEST_F(QstEvaluatorTest, PieceFeaturesBlackToMove) {
  // Same as above but black to move. Boards should be flipped and swapped 
  // (black for features[0], white for features[1], instead of vice versa).
  Position pos("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2");
  QstEvaluator qstEvaluator;
  std::vector<Bitboard> features;
  qstEvaluator.get_features<Color::BLACK>(pos, &features);
  EXPECT_BB_EQ(features[0], flip_vertically(BLACK_PAWNS_START));
  EXPECT_BB_EQ(features[1], flip_vertically(WHITE_PAWNS_AFTER_E4));
}
