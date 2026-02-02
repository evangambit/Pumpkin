#include "../../game/geometry.h"
#include "../../game/utils.h"
#include "../../game/Position.h"
#include "../../game/movegen/movegen.h"
#include "../../game/movegen/sliding.h"
#include "QstEvaluator.h"
#include "../nnue/sharded_matrix.h"

#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <bit>

/*
 * This tool converts a text file of chess positions into sharded binary matrices
 * for training the QstEvaluator.
 * 
 * Usage: ./make_tables_qst <input_file> <output_prefix>
 */

using namespace ChessEngine;
using WriterI16 = ShardedMatrix::Writer<int16_t>;
using WriterI8 = ShardedMatrix::Writer<int8_t>;
using WriterB = ShardedMatrix::Writer<bool>;
constexpr size_t NUM_BITBOARDS = 54;

void process(
  const std::vector<std::string>& line,
  WriterB& qstInputWriter,
  WriterI16& evalWriter,
  WriterI8& turnWriter,
  WriterI8& pieceCountWriter
  ) {
  if (line.size() != 4) {
    std::cout << "error: line has " << line.size() << " elements" << std::endl;
    return;
  }

  Position pos(line[0]);
  std::vector<Bitboard> features;
  if (pos.turn_ == Color::WHITE) {
    QstEvaluator::get_features<Color::WHITE>(pos, &features);
  } else {
    QstEvaluator::get_features<Color::BLACK>(pos, &features);
  }
  
  if (features.size() != NUM_BITBOARDS) {
      std::cerr << "Error: Expected " << NUM_BITBOARDS << " features, got " << features.size() << std::endl;
      return;
  }

  bool qstFeatures[NUM_BITBOARDS * 64];
  for (size_t i = 0; i < NUM_BITBOARDS; ++i) {
    for (size_t j = 0; j < 64; ++j) {
      qstFeatures[i * 64 + j] = (features[i] >> j) & 1;
    }
  }
  qstInputWriter.write_row(qstFeatures);

  int16_t wdl[3] = {
    int16_t(std::stoi(line[1])),
    int16_t(std::stoi(line[2])),
    int16_t(std::stoi(line[3])),
  };
  evalWriter.write_row(wdl);

  int8_t turn = pos.turn_ == Color::WHITE ? 1 : -1;
  turnWriter.write_row(&turn);

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
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input> <output>" << std::endl;
    return 1;
  }

  const std::string inpath = argv[1];
  const std::string outpath = argv[2];

  std::ifstream infile(inpath);
  if (!infile.is_open()) {
    std::cerr << "Could not open input file: " << inpath << std::endl;
    return 1;
  }

  WriterB qstInputWriter(outpath + "-qst", { NUM_BITBOARDS * 64 });
  WriterI16 evalWriter(outpath + "-eval", { 3 });
  WriterI8 turnWriter(outpath + "-turn", { 1 });
  WriterI8 pieceCountWriter(outpath + "-piece-counts", { 10 });

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  size_t counter = 0;
  std::string line;
  while (std::getline(infile, line)) {
    if (line == "") {
      continue;
    }
    std::vector<std::string> parts = split(line, '|');
    process(parts, qstInputWriter, evalWriter, turnWriter, pieceCountWriter);

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
      break;
    }
  }

  std::cout << "Completed " << counter << " positions." << std::endl;

  return 0;
}
