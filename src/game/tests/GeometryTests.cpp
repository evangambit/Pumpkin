#include <gtest/gtest.h>

#include <vector>

#include "../utils.h"
#include "../geometry.h"

#define EXPECT_BB_EQ(actual, expected) \
  EXPECT_TRUE((actual) == (expected)) \
    << "Bitboards differ:\n" \
    << "Actual:\n" << bstr(actual) \
    << "Expected:\n" << bstr(expected)

    namespace ChessEngine {

class GeometryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_geometry();
  }
};

TEST_F(GeometryTest, NorthFill) {
  EXPECT_BB_EQ(northFill(bb(SA2) | bb(SC4)), bb(SA2) | bb(SA3) | bb(SA4) | bb(SA5) | bb(SA6) | bb(SA7) | bb(SA8) | bb(SC4) | bb(SC5) | bb(SC6) | bb(SC7) | bb(SC8));
}

TEST_F(GeometryTest, SouthFill) {
  EXPECT_BB_EQ(southFill(bb(SA7) | bb(SC5)), bb(SA1) | bb(SA2) | bb(SA3) | bb(SA4) | bb(SA5) | bb(SA6) | bb(SA7) | bb(SC1) | bb(SC2) | bb(SC3) | bb(SC4) | bb(SC5));
}

TEST_F(GeometryTest, Shift) {
  // Simple shifts
  EXPECT_BB_EQ(shift<Direction::NORTH>(bb(SE4)), bb(SE5));
  EXPECT_BB_EQ(shift<Direction::SOUTH>(bb(SE4)), bb(SE3));
  EXPECT_BB_EQ(shift<Direction::EAST>(bb(SE4)), bb(SF4));
  EXPECT_BB_EQ(shift<Direction::WEST>(bb(SE4)), bb(SD4));
  // Masking
  EXPECT_BB_EQ(shift<Direction::NORTH>(bb(SA8)), kEmptyBitboard);
  EXPECT_BB_EQ(shift<Direction::WEST>(bb(SA8)), kEmptyBitboard);
  EXPECT_BB_EQ(shift<Direction::EAST>(bb(SH1)), kEmptyBitboard);
  EXPECT_BB_EQ(shift<Direction::SOUTH>(bb(SH1)), kEmptyBitboard);
  // Full board tests.
  EXPECT_BB_EQ(shift_ew<Direction::EAST>(kUniverse, 1), kUniverse & ~kFiles[0]);
  EXPECT_BB_EQ(shift_ew<Direction::WEST>(kUniverse, 1), kUniverse & ~kFiles[7]);
  EXPECT_BB_EQ(shift<Direction::NORTH>(kUniverse), kUniverse & ~kRanks[7]);
  EXPECT_BB_EQ(shift<Direction::SOUTH>(kUniverse), kUniverse & ~kRanks[0]);
}

TEST_F(GeometryTest, Shift_EW) {
  // Simple shifts
  EXPECT_BB_EQ(shift_ew<Direction::EAST>(bb(SA2), 3), bb(SD2));
  EXPECT_BB_EQ(shift_ew<Direction::WEST>(bb(SD2), 3), bb(SA2));
  // Masking
  EXPECT_BB_EQ(shift_ew<Direction::EAST>(bb(SE2) | bb(SF2), 3), bb(SH2));
  EXPECT_BB_EQ(shift_ew<Direction::WEST>(bb(SC2) | bb(SD2), 3), bb(SA2));
  // Full board tests.
  EXPECT_BB_EQ(shift_ew<Direction::EAST>(kUniverse, 1), kUniverse & ~kFiles[0]);
  EXPECT_BB_EQ(shift_ew<Direction::EAST>(kUniverse, 2), kUniverse & ~kFiles[0] & ~kFiles[1]);
  EXPECT_BB_EQ(shift_ew<Direction::WEST>(kUniverse, 1), kUniverse & ~kFiles[7]);
  EXPECT_BB_EQ(shift_ew<Direction::WEST>(kUniverse, 2), kUniverse & ~kFiles[7] & ~kFiles[6]);
}

TEST_F(GeometryTest, FlipVertically) {
  EXPECT_BB_EQ(flip_vertically(bb(SA2) | bb(SB3)), bb(SA7) | bb(SB6));
}

TEST_F(GeometryTest, FlipHorizontally) {
  EXPECT_BB_EQ(flip_horizontally(bb(SA2) | bb(SB3)), bb(SH2) | bb(SG3));
}

}  // namespace ChessEngine
