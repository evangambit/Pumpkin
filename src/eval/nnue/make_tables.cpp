#include "../../game/geometry.h"
#include "../../game/utils.h"
#include "../../game/Position.h"
#include "../../game/movegen/movegen.h"
#include "../../game/movegen/sliding.h"
#include "nnue.h"
#include "sharded_matrix.h"

#include <thread>

// sqlite3 data/de6-md2/db.sqlite3 'select * from positions' > /tmp/out.txt
// shuf /tmp/out.txt > /tmp/out.shuf.txt
// ./make_tables /tmp/out.shuf.txt data/de6-md2/tables

using namespace ChessEngine;
using WriterF32 = ShardedMatrix::Writer<float>;
using WriterB = ShardedMatrix::Writer<bool>;
using WriterI8 = ShardedMatrix::Writer<int8_t>;
using WriterI16 = ShardedMatrix::Writer<int16_t>;

void process(
  const std::vector<std::string>& line,
  WriterI16& sparseNnueInputWriter,
  WriterI16& evalWriter,
  WriterI8& turnWriter,
  WriterI8& pieceCountWriter
  ) {
  if (line.size() != 4) {
    std::cout << "error: line has " << line.size() << " elements" << std::endl;
    return;
  }

  Position pos(line[0]);
  NNUE::Features features = NNUE::pos2features(pos);
  sparseNnueInputWriter.write_row(features.onIndices);

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

std::string get_shard_name(size_t n) {
  // return string with 0 padding
  return std::string(5 - std::to_string(n).length(), '0') + std::to_string(n);
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
  WriterI16 sparseNnueInputWriter(outpath + "-nnue", { NNUE::MAX_NUM_ONES_IN_INPUT });
  WriterI16 evalWriter(outpath + "-eval", { 3 });
  WriterI8 turnWriter(outpath + "-turn", { 1 });
  WriterI8 pieceCountWriter(outpath + "-piece-counts", { 10 });

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  size_t counter = 0;
  std::string line;
  if (!infile.is_open()) {
    std::cerr << "Could not open file: " << inpath << std::endl;
    return 1;
  }
  while (std::getline(infile, line)) {
    if (line == "") {
      continue;
    }
    std::vector<std::string> parts = split(line, '|');
    process(parts, sparseNnueInputWriter, evalWriter, turnWriter, pieceCountWriter);

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
    }
  }

  return 0;
}