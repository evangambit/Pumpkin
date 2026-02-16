#include "../../game/Geometry.h"
#include "../../game/Utils.h"
#include "../../game/Position.h"
#include "../../game/movegen/movegen.h"
#include "../../game/movegen/sliding.h"
#include "QstEvaluator.h"
#include "../nnue/ShardedMatrix.h"
#include "../nnue/Nnue.h"

#include <gflags/gflags.h>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <bit>

DEFINE_string(output, "", "Output prefix for the sharded matrices");
DEFINE_bool(emit_qst, false, "Whether to emit QST features");
DEFINE_bool(emit_nnue, false, "Whether to emit NNUE features");

/*
 * This tool converts a text file of chess positions into sharded binary matrices
 * for training the QstEvaluator.
 * 
 * Usage: cat <input_file> | ./make_tables_qst --output=<output_prefix>
 */

using namespace ChessEngine;
using WriterI16 = ShardedMatrix::Writer<int16_t>;
using WriterI8 = ShardedMatrix::Writer<int8_t>;
using WriterB = ShardedMatrix::Writer<bool>;

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

void process(
  const std::vector<std::string>& line,
  WriterB& qstInputWriter,
  WriterI16& evalWriter,
  WriterI8& pieceCountWriter,
  QstEvaluator& qstEvaluator,
  WriterI16& sparseNnueInputWriter
  ) {
  if (line.size() != 4 && line.size() != 2) {
    std::cout << "error: line has " << line.size() << " elements" << std::endl;
    return;
  }

  // If line has 2 elements, it's in the format: fen | score
  // If line has 4 elements, it's in the format: fen | win | draw | loss

  Position pos(line[0]);

  if (FLAGS_emit_qst) {
    std::vector<Bitboard> features;
    features.reserve(Q_NUM_FEATURES);
    if (pos.turn_ == Color::WHITE) {
      qstEvaluator.get_features<Color::WHITE>(pos, &features);
    } else {
      qstEvaluator.get_features<Color::BLACK>(pos, &features);
    }
    
    if (features.size() != Q_NUM_FEATURES) {
      std::cerr << "Error: Expected " << Q_NUM_FEATURES << " features, got " << features.size() << std::endl;
      return;
    }

    bool qstFeatures[Q_NUM_FEATURES * 64];
    for (size_t i = 0; i < Q_NUM_FEATURES; ++i) {
      for (size_t j = 0; j < 64; ++j) {
        qstFeatures[i * 64 + j] = (features[i] >> j) & 1;
      }
    }
    qstInputWriter.write_row(qstFeatures);
  }

  if (FLAGS_emit_nnue) {
    NNUE::Features features = NNUE::pos2features(pos);
    if (pos.turn_ == Color::BLACK) {
      features.flip_();
    }
    sparseNnueInputWriter.write_row(features.onIndices);
  }

  int16_t wdl[3];
  if (line.size() == 4) {
    wdl[0] = int16_t(std::stoi(line[1]));
    wdl[1] = int16_t(std::stoi(line[2]));
    wdl[2] = int16_t(std::stoi(line[3]));
  } else {
    float score = std::stof(line[1]);
    /*
    Quick and dirty conversion from score to WDL.

    Assume
    p( win | score = 1.0) = 0.48
    p( win | score = 0.0) = 0.21
    (From some past analysis)

    Then the logistic function parameters are:
    p(win) = sigmoid(1.25 x - 1.33)
    */
    float winRate = sigmoid(1.25f * score - 1.33f);
    float loseRate = sigmoid(-1.25f * score - 1.33f);
    wdl[0] = int16_t(std::round(winRate * 1000.0f));
    wdl[2] = int16_t(std::round(loseRate * 1000.0f));
    wdl[1] = 1000 - wdl[0] - wdl[2];
  }
  evalWriter.write_row(wdl);

  int8_t pieceCounts[10];
  pieceCounts[0] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]);
  pieceCounts[1] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]);
  pieceCounts[2] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
  pieceCounts[3] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]);
  pieceCounts[4] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
  pieceCounts[5] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]);
  pieceCounts[6] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]);
  pieceCounts[7] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
  pieceCounts[8] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]);
  pieceCounts[9] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);
  pieceCountWriter.write_row(pieceCounts);
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  if (FLAGS_output.empty()) {
    std::cerr << "Usage: cat <input> | " << argv[0] << " --output=<output_prefix>" << std::endl;
    return 1;
  }

  const std::string outpath = FLAGS_output;

  WriterB qstInputWriter(outpath + "-qst", { Q_NUM_FEATURES * 64 });
  WriterI16 evalWriter(outpath + "-eval", { 3 });
  WriterI8 pieceCountWriter(outpath + "-piece-counts", { 10 });
  WriterI16 sparseNnueInputWriter(outpath + "-nnue", { NNUE::MAX_NUM_ONES_IN_INPUT });

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  QstEvaluator qstEvaluator;

  size_t counter = 0;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "") {
      continue;
    }
    std::vector<std::string> parts = split(line, '|');
    process(parts, qstInputWriter, evalWriter, pieceCountWriter, qstEvaluator, sparseNnueInputWriter);

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
    }
  }

  std::cout << "Completed " << counter << " positions." << std::endl;

  return 0;
}
