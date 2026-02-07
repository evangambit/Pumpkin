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

TEST_F(QstEvaluatorTest, ShiftToDestination) {
  const Bitboard fourCorners = bb(SA1) | bb(SA8) | bb(SH1) | bb(SH8);
  EXPECT_BB_EQ(shiftToDestination<SafeSquare::SE4>(SafeSquare::SE4, fourCorners), fourCorners);  // shift to same square should do nothing
  EXPECT_BB_EQ(shiftToDestination<SafeSquare::SE4>(SafeSquare::SE3, fourCorners), bb(SA2) | bb(SH2));  // shift up 1 (E3 -> E4)
  EXPECT_BB_EQ(shiftToDestination<SafeSquare::SE4>(SafeSquare::SE5, fourCorners), bb(SA7) | bb(SH7));  // shift down 1 (E5 -> E4)
  EXPECT_BB_EQ(shiftToDestination<SafeSquare::SE4>(SafeSquare::SD4, fourCorners), bb(SB1) | bb(SB8));  // shift right 1 (D4 -> E4)
  EXPECT_BB_EQ(shiftToDestination<SafeSquare::SE4>(SafeSquare::SF4, fourCorners), bb(SG1) | bb(SG8));  // shift left 1 (F4 -> E4)
}

TEST_F(QstEvaluatorTest, FeatureCountMatchesExpected) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  QstEvaluator qstEvaluator;
  std::vector<Bitboard> features;
  qstEvaluator.get_features<Color::WHITE>(pos, &features);
  EXPECT_EQ(features.size(), Q_NUM_FEATURES);
}

constexpr Bitboard WHITE_PAWNS_AFTER_E4 = bb(SA2) | bb(SB2) | bb(SC2) | bb(SD2) | bb(SE4) | bb(SF2) | bb(SG2) | bb(SH2);
constexpr Bitboard BLACK_PAWNS_START = bb(SA7) | bb(SB7) | bb(SC7) | bb(SD7) | bb(SE7) | bb(SF7) | bb(SG7) | bb(SH7);

TEST_F(QstEvaluatorTest, Symmetry) {
  // Some random positions.
  std::string fens[] = {
    "k6r/1q2b3/p3P3/Pp2Q1pp/1P6/4P1P1/4N3/2R3K1 b - - 1 33",
    "2r2rk1/p4ppp/1p2pn2/2pp4/q2P1P2/2P2N1P/PP2QPP1/2R1R1K1 b - - 3 15",
    "1r4k1/1p1nrppb/p2p1q2/P1pP1P1p/Q1P1P3/5N1N/6PP/1R3R1K b - - 4 34",
    "4b3/8/p3p2k/Pp1pPp2/5Kp1/1PP3P1/2B5/8 b - - 46 81",
    "6rk/3q2pb/5p1p/3R1P1N/r2p3P/3P1Q2/6PK/1R6 b - - 7 36",
    "8/6b1/1N1nkp2/p1p5/P1P2PP1/3B1K2/8/8 w - - 20 49",
    "2r1r1k1/p2b1q2/3p1p1p/1p1P1p2/2nN1NnP/4P3/P3QPB1/2R2RK1 b - - 0 24",
    "r2qkb1r/pb1p1ppp/2n1pn2/1pp5/4P1P1/3P1N1P/PPP2PB1/RNBQK2R w KQkq - 0 5",
    "r2qkbnr/1p3p2/p1n5/2Pp3p/6b1/2N1PN2/PPB2PP1/R1BQK2R b KQkq - 0 9",
    "5r2/pr1b2pk/3p1nqp/2pP1p1n/PpP1pP1P/1P2P1PB/1B1N1Q1K/R5R1 w - - 6 31",
  };
  for (const std::string& fen : fens) {
    Position pos(fen);
    QstEvaluator qstEvaluator;
    std::vector<Bitboard> featuresWhite;
    std::vector<Bitboard> featuresBlack;
    if (pos.turn_ == Color::WHITE) {
      qstEvaluator.get_features<Color::WHITE>(pos, &featuresWhite);
      make_nullmove<Color::WHITE>(&pos);
      qstEvaluator.get_features<Color::BLACK>(pos, &featuresBlack);
    } else {
      qstEvaluator.get_features<Color::BLACK>(pos, &featuresBlack);
      make_nullmove<Color::BLACK>(&pos);
      qstEvaluator.get_features<Color::WHITE>(pos, &featuresWhite);
    }

    for (size_t i = 0; i < Q_NUM_FEATURES; i += 2) {
      EXPECT_BB_EQ(featuresWhite[i], flip_vertically(featuresBlack[i + 1])) << "Feature index: " << i;
      EXPECT_BB_EQ(featuresWhite[i + 1], flip_vertically(featuresBlack[i])) << "Feature index: " << i + 1;
    }
  }
}

TEST_F(QstEvaluatorTest, Symmetry2) {
  // Some random positions.
  Position whitePos("4b3/8/p3p2k/Pp1pPp2/5Kp1/1PP3P1/2B5/8 w - - 46 81");
  QstEvaluator qstEvaluator;
  std::vector<Bitboard> featuresWhite;
  qstEvaluator.get_features<Color::WHITE>(whitePos, &featuresWhite);

  Position blackPos("4b3/8/p3p2k/Pp1pPp2/5Kp1/1PP3P1/2B5/8 b - - 46 81");
  std::vector<Bitboard> featuresBlack;
  qstEvaluator.get_features<Color::BLACK>(blackPos, &featuresBlack);

  EXPECT_BB_EQ(featuresWhite[Q_BAD_FOR_PAWN_US], flip_vertically(featuresBlack[Q_BAD_FOR_PAWN_THEM]));
}


// TEST_F(QstEvaluatorTest, PieceFeatures) {
//   // 1. e4 Nc6
//   Position pos("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2");
//   QstEvaluator qstEvaluator;
//   std::vector<Bitboard> features;
//   qstEvaluator.get_features<Color::WHITE>(pos, &features);
//   EXPECT_BB_EQ(features[Q_PAWNS_US], WHITE_PAWNS_AFTER_E4);
//   EXPECT_BB_EQ(features[Q_PAWNS_THEM], BLACK_PAWNS_START);
// }

// TEST_F(QstEvaluatorTest, PieceFeaturesBlackToMove) {
//   // Same as above but black to move. Boards should be flipped and swapped 
//   // (black for features[0], white for features[1], instead of vice versa).
//   Position pos("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2");
//   QstEvaluator qstEvaluator;
//   std::vector<Bitboard> features;
//   qstEvaluator.get_features<Color::BLACK>(pos, &features);
//   EXPECT_BB_EQ(features[Q_PAWNS_US], flip_vertically(BLACK_PAWNS_START));
//   EXPECT_BB_EQ(features[Q_PAWNS_THEM], flip_vertically(WHITE_PAWNS_AFTER_E4));
// }
