#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>

#include "../nnue.h"
#include "../NnueEvaluator.h"
#include "../../../game/Position.h"
#include "../../../game/movegen/movegen.h"

using namespace NNUE;
using namespace ChessEngine;

class NNUETest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

// Test randn() produces reasonable values
TEST_F(NNUETest, RandnDistribution) {
  const int num_samples = 10000;
  double sum = 0;
  double sum_squares = 0;
  
  for (int i = 0; i < num_samples; ++i) {
    double val = randn(1.0);
    sum += val;
    sum_squares += val * val;
  }
  
  double mean = sum / num_samples;
  double variance = sum_squares / num_samples - mean * mean;
  
  // Mean should be close to 0
  EXPECT_NEAR(mean, 0.0, 0.1);
  // Variance should be close to 1 (stddev = 1.0)
  EXPECT_NEAR(variance, 1.0, 0.2);
}

// Test randn() with custom stddev
TEST_F(NNUETest, RandnCustomStddev) {
  const int num_samples = 1000;
  double stddev = 2.5;
  double sum_squares = 0;
  
  for (int i = 0; i < num_samples; ++i) {
    double val = randn(stddev);
    sum_squares += val * val;
  }
  
  double variance = sum_squares / num_samples;
  double actual_stddev = std::sqrt(variance);
  
  EXPECT_NEAR(actual_stddev, stddev, stddev * 0.3);
}

// Test Features struct initialization
TEST_F(NNUETest, FeaturesInitialization) {
  Features features;
  
  EXPECT_EQ(features.length, 0);
  for (int i = 0; i < MAX_NUM_ONES_IN_INPUT; ++i) {
    EXPECT_EQ(features[i], SpecialFeatures::INPUT_DIM);
  }
}

// Test Features addFeature
TEST_F(NNUETest, FeaturesAddFeature) {
  Features features;
  
  features.addFeature(0);
  EXPECT_EQ(features.length, 1);
  EXPECT_EQ(features[0], 0);
  
  features.addFeature(100);
  EXPECT_EQ(features.length, 2);
  EXPECT_EQ(features[1], 100);
  
  features.addFeature(767);
  EXPECT_EQ(features.length, 3);
  EXPECT_EQ(features[2], 767);
}

// Test feature_index function
TEST_F(NNUETest, FeatureIndex) {
  // Test a few combinations
  int16_t idx = feature_index(SafeColoredPiece(0), SafeSquare(0));
  EXPECT_EQ(idx, 0);  // 0 * 64 + 0
  
  idx = feature_index(SafeColoredPiece(0), SafeSquare(63));
  EXPECT_EQ(idx, 63);  // 0 * 64 + 63
  
  idx = feature_index(SafeColoredPiece(1), SafeSquare(0));
  EXPECT_EQ(idx, 64);  // 1 * 64 + 0
  
  idx = feature_index(SafeColoredPiece(11), SafeSquare(63));
  EXPECT_EQ(idx, 767);  // 11 * 64 + 63
}

// Test flip_feature_index
TEST_F(NNUETest, FlipFeatureIndex) {
  // Flipping twice should give a different index (color swap + rank flip)
  int16_t original = 0;  // White pawn on a1
  int16_t flipped = flip_feature_index(original);
  
  // Should not be the same
  EXPECT_NE(flipped, original);
  
  // Test a known case
  int16_t white_pawn_a1 = feature_index(SafeColoredPiece(0), SafeSquare(0));
  int16_t flipped_idx = flip_feature_index(white_pawn_a1);
  // Black's piece should be involved, and square should be flipped vertically
  EXPECT_GT(flipped_idx, 255);  // Black pieces start at higher indices
}

// Test pos2features with starting position
TEST_F(NNUETest, Pos2FeaturesStartingPosition) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  Features features = pos2features(pos);
  
  // Starting position should have 32 pieces + 4 (for castling rights).
  EXPECT_EQ(features.length, 36);
  
  // All feature indices should be in valid range
  for (int i = 0; i < features.length; ++i) {
    EXPECT_GE(features[i], 0);
    EXPECT_LT(features[i], SpecialFeatures::INPUT_DIM);
  }
}

// Test pos2features with castling rights
TEST_F(NNUETest, Pos2FeaturesCastlingRights) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  Features features = pos2features(pos);
  
  bool has_white_kingside = false;
  bool has_white_queenside = false;
  bool has_black_kingside = false;
  bool has_black_queenside = false;
  
  for (int i = 0; i < features.length; ++i) {
    if (features[i] == SpecialFeatures::WHITE_KINGSIDE_CASTLING_RIGHT) {
      has_white_kingside = true;
    }
    if (features[i] == SpecialFeatures::WHITE_QUEENSIDE_CASTLING_RIGHT) {
      has_white_queenside = true;
    }
    if (features[i] == SpecialFeatures::BLACK_KINGSIDE_CASTLING_RIGHT) {
      has_black_kingside = true;
    }
    if (features[i] == SpecialFeatures::BLACK_QUEENSIDE_CASTLING_RIGHT) {
      has_black_queenside = true;
    }
  }
  
  EXPECT_TRUE(has_white_kingside);
  EXPECT_TRUE(has_white_queenside);
  EXPECT_TRUE(has_black_kingside);
  EXPECT_TRUE(has_black_queenside);
}

// Test pos2features without castling rights
TEST_F(NNUETest, Pos2FeaturesNoCastlingRights) {
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
  
  Features features = pos2features(pos);
  
  bool has_castling = false;
  for (int i = 0; i < features.length; ++i) {
    if (features[i] == SpecialFeatures::WHITE_KINGSIDE_CASTLING_RIGHT ||
        features[i] == SpecialFeatures::WHITE_QUEENSIDE_CASTLING_RIGHT ||
        features[i] == SpecialFeatures::BLACK_KINGSIDE_CASTLING_RIGHT ||
        features[i] == SpecialFeatures::BLACK_QUEENSIDE_CASTLING_RIGHT) {
      has_castling = true;
    }
  }
  
  EXPECT_FALSE(has_castling);
}

// Test NNUE initialization
TEST_F(NNUETest, NnueInitialization) {
  Nnue nnue;
  
  // All x values should be false
  for (int i = 0; i < INPUT_DIM; ++i) {
    EXPECT_FALSE(nnue.x[i]);
  }
  
  // Accumulators should be zero
  EXPECT_TRUE(nnue.whiteAcc.isZero());
  EXPECT_TRUE(nnue.blackAcc.isZero());
}

// Test NNUE randn_
TEST_F(NNUETest, NnueRandn) {
  Nnue nnue;
  nnue.randn_();
  
  // After random initialization, matrices should not be all zero
  EXPECT_FALSE(nnue.layer1.isZero());
  EXPECT_FALSE(nnue.layer2.isZero());
  EXPECT_FALSE(nnue.layer3.isZero());
  
  // Embedding weights should be initialized
  bool has_nonzero_emb = false;
  for (int i = 0; i < INPUT_DIM; ++i) {
    if (!nnue.embWeights[i].isZero()) {
      has_nonzero_emb = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero_emb);
}

// Test NNUE increment
TEST_F(NNUETest, NnueIncrement) {
  Nnue nnue;
  nnue.randn_();
  
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> initial_white = nnue.whiteAcc;
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> initial_black = nnue.blackAcc;
  
  size_t index = 100;
  nnue.increment(index);
  
  EXPECT_TRUE(nnue.x[index]);
  EXPECT_NE((nnue.whiteAcc - initial_white).isZero(), true);
  EXPECT_NE((nnue.blackAcc - initial_black).isZero(), true);
}

// Test NNUE decrement
TEST_F(NNUETest, NnueDecrement) {
  Nnue nnue;
  nnue.randn_();
  
  size_t index = 100;
  nnue.increment(index);
  
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> after_incr = nnue.whiteAcc;
  
  nnue.decrement(index);
  
  EXPECT_FALSE(nnue.x[index]);
  EXPECT_TRUE(nnue.whiteAcc.isZero());
  EXPECT_TRUE(nnue.blackAcc.isZero());
}

// Test NNUE increment/decrement symmetry
TEST_F(NNUETest, NnueIncrementDecrementSymmetry) {
  Nnue nnue;
  nnue.randn_();
  
  size_t index = 50;
  nnue.increment(index);
  nnue.decrement(index);
  
  // After increment then decrement, should be back to zero
  EXPECT_TRUE(nnue.whiteAcc.isZero());
  EXPECT_TRUE(nnue.blackAcc.isZero());
  EXPECT_FALSE(nnue.x[index]);
}

// Test NNUE clear_accumulator
TEST_F(NNUETest, NnueClearAccumulator) {
  Nnue nnue;
  nnue.randn_();
  
  size_t index = 100;
  nnue.increment(index);
  EXPECT_FALSE(nnue.whiteAcc.isZero());
  
  nnue.clear_accumulator();
  
  EXPECT_TRUE(nnue.whiteAcc.isZero());
  EXPECT_TRUE(nnue.blackAcc.isZero());
  // x should still be set
  EXPECT_TRUE(nnue.x[index]);
}

// Test NNUE compute_acc_from_scratch
TEST_F(NNUETest, NnueComputeAccFromScratch) {
  Nnue nnue;
  nnue.randn_();
  
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  nnue.compute_acc_from_scratch(pos);
  
  // After computing from scratch, accumulators should be filled
  EXPECT_FALSE(nnue.whiteAcc.isZero());
  EXPECT_FALSE(nnue.blackAcc.isZero());
  
  // x array should reflect the board position
  bool has_features = false;
  for (int i = 0; i < INPUT_DIM; ++i) {
    if (nnue.x[i]) {
      has_features = true;
      break;
    }
  }
  EXPECT_TRUE(has_features);
}

// Test NNUE compute_acc_from_scratch empty board
TEST_F(NNUETest, NnueComputeAccFromScratchEmptyBoard) {
  Nnue nnue;
  nnue.randn_();
  
  Position pos("8/8/8/8/8/8/8/8 w - - 0 1");
  
  nnue.compute_acc_from_scratch(pos);
  
  // Empty board means no piece features, only castling (if any)
  EXPECT_TRUE(nnue.whiteAcc.isZero());
  EXPECT_TRUE(nnue.blackAcc.isZero());
}

// Test NNUE forward with WHITE to move
TEST_F(NNUETest, NnueForwardWhite) {
  Nnue nnue;
  nnue.randn_();
  
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  nnue.compute_acc_from_scratch(pos);
  int16_t *output = nnue.forward(Color::WHITE);
  
  // Output should be non-null
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output, nnue.output.data());
  
  // Output dimension should match
  EXPECT_EQ(nnue.output.cols(), OUTPUT_DIM);
}

// Test NNUE forward with BLACK to move
TEST_F(NNUETest, NnueForwardBlack) {
  Nnue nnue;
  nnue.randn_();
  
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
  
  nnue.compute_acc_from_scratch(pos);
  int16_t *output = nnue.forward(Color::BLACK);
  
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output, nnue.output.data());
}

// Test NNUE clone
TEST_F(NNUETest, NnueClone) {
  Nnue nnue;
  nnue.randn_();
  
  auto cloned = nnue.clone();
  
  // Check that it's not null and is a different object
  EXPECT_NE(cloned.get(), &nnue);
  
  // Check that embedding weights are copied
  for (int i = 0; i < INPUT_DIM; ++i) {
    EXPECT_TRUE((cloned->embWeights[i] - nnue.embWeights[i]).isZero(1e-10));
  }
  
  // Check that layers are copied
  EXPECT_TRUE((cloned->layer1 - nnue.layer1).isZero(1e-10));
  EXPECT_TRUE((cloned->layer2 - nnue.layer2).isZero(1e-10));
  EXPECT_TRUE((cloned->layer3 - nnue.layer3).isZero(1e-10));
  
  // Check that biases are copied
  EXPECT_TRUE((cloned->bias1 - nnue.bias1).isZero(1e-10));
  EXPECT_TRUE((cloned->bias2 - nnue.bias2).isZero(1e-10));
  EXPECT_TRUE((cloned->bias3 - nnue.bias3).isZero(1e-10));
}

// Test constant values
TEST_F(NNUETest, ConstantValues) {
  EXPECT_EQ(SCALE_SHIFT, 6);
  EXPECT_EQ(EMBEDDING_DIM, 1024);
  EXPECT_EQ(HIDDEN1_DIM, 256);
  EXPECT_EQ(HIDDEN2_DIM, 64);
  EXPECT_EQ(OUTPUT_DIM, 16);
  EXPECT_EQ(MAX_NUM_ONES_IN_INPUT, 32 + 4);
  EXPECT_EQ(SpecialFeatures::INPUT_DIM, 768);
}

// Test SpecialFeatures enum values
TEST_F(NNUETest, SpecialFeaturesEnumValues) {
  EXPECT_EQ(SpecialFeatures::WHITE_KINGSIDE_CASTLING_RIGHT, 0);
  EXPECT_EQ(SpecialFeatures::WHITE_QUEENSIDE_CASTLING_RIGHT, 1);
  EXPECT_EQ(SpecialFeatures::BLACK_KINGSIDE_CASTLING_RIGHT, 440);
  EXPECT_EQ(SpecialFeatures::BLACK_QUEENSIDE_CASTLING_RIGHT, 441);
}

// Test multiple increments and decrements
TEST_F(NNUETest, NnueMultipleIncrementDecrement) {
  Nnue nnue;
  nnue.randn_();
  
  std::vector<size_t> indices = {10, 50, 100, 200, 300};
  
  for (size_t idx : indices) {
    nnue.increment(idx);
    EXPECT_TRUE(nnue.x[idx]);
  }
  
  Eigen::Matrix<int16_t, 1, EMBEDDING_DIM> after_increments = nnue.whiteAcc;
  EXPECT_FALSE(after_increments.isZero());
  
  for (size_t idx : indices) {
    nnue.decrement(idx);
    EXPECT_FALSE(nnue.x[idx]);
  }
  
  EXPECT_TRUE(nnue.whiteAcc.isZero());
  EXPECT_TRUE(nnue.blackAcc.isZero());
}

// Test empty Features
TEST_F(NNUETest, EmptyFeatures) {
  Features features;
  EXPECT_EQ(features.length, 0);
  
  // Accessing elements should return INPUT_DIM
  for (int i = 0; i < MAX_NUM_ONES_IN_INPUT; ++i) {
    EXPECT_EQ(features[i], SpecialFeatures::INPUT_DIM);
  }
}

// Test flip_feature_index consistency
TEST_F(NNUETest, FlipFeatureIndexConsistency) {
  // Test that flipping different squares gives different results
  int16_t idx1 = flip_feature_index(0);
  int16_t idx2 = flip_feature_index(1);
  int16_t idx3 = flip_feature_index(64);
  
  EXPECT_NE(idx1, idx2);
  EXPECT_NE(idx1, idx3);
  EXPECT_NE(idx2, idx3);
  
  // All should be in valid range
  for (int i = 0; i < SpecialFeatures::INPUT_DIM; ++i) {
    int16_t flipped = flip_feature_index(i);
    EXPECT_GE(flipped, 0);
    EXPECT_LT(flipped, SpecialFeatures::INPUT_DIM);
  }
}

// Test accumulator state after compute_acc_from_scratch
TEST_F(NNUETest, AccumulatorStateAfterCompute) {
  Nnue nnue;
  nnue.randn_();
  
  Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  nnue.compute_acc_from_scratch(pos);
  
  auto white_acc_copy = nnue.whiteAcc;
  auto black_acc_copy = nnue.blackAcc;
  
  // Computing again should give the same result
  nnue.compute_acc_from_scratch(pos);
  
  EXPECT_TRUE((nnue.whiteAcc - white_acc_copy).isZero());
  EXPECT_TRUE((nnue.blackAcc - black_acc_copy).isZero());
}

TEST_F(NNUETest, AccumulatorIncrementallyUpdated) {
    std::shared_ptr<Nnue> nnue = std::make_shared<Nnue>();
    nnue->randn_();
    std::shared_ptr<NnueEvaluator> evaluator = std::make_shared<NnueEvaluator>(nnue);
    
    Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    pos.set_listener(evaluator);

    std::vector<int16_t> initial_input(nnue->x, nnue->x + INPUT_DIM);
    
    make_move<Color::WHITE>(&pos, Move::fromUci("e2e4"));

    std::vector<int16_t> after_e4_input(nnue->x, nnue->x + INPUT_DIM);

    make_move<Color::BLACK>(&pos, Move::fromUci("e7e5"));

    std::vector<int16_t> after_e5_input(nnue->x, nnue->x + INPUT_DIM);

    undo<Color::BLACK>(&pos);

    std::vector<int16_t> after_undo_e5_input(nnue->x, nnue->x + INPUT_DIM);

    undo<Color::WHITE>(&pos);

    std::vector<int16_t> after_undo_e4_input(nnue->x, nnue->x + INPUT_DIM);

    EXPECT_EQ(initial_input, after_undo_e4_input);
    EXPECT_NE(initial_input, after_e4_input);
    EXPECT_NE(initial_input, after_e5_input);

    EXPECT_EQ(after_e4_input, after_undo_e5_input);
    EXPECT_NE(after_e4_input, after_e5_input);
}