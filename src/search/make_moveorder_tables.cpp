#include "../game/Geometry.h"
#include "../game/Utils.h"
#include "../game/Position.h"
#include "../game/Move.h"
#include "../game/movegen/movegen.h"
#include "../eval/nnue/ShardedMatrix.h"

#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>

/*
 * This tool converts a text file of chess positions into sharded binary matrices
 * for training move ordering.
 * 
 * Input format: <fen>|score|move1|move2|move3
 * 
 * Usage: ./make_moveorder_tables <input_file> <output_prefix> [limit]
 */

using namespace ChessEngine;
using WriterB = ShardedMatrix::Writer<bool>;
using WriterI8 = ShardedMatrix::Writer<int8_t>;
using WriterI16 = ShardedMatrix::Writer<int16_t>;

void process(
  const std::vector<std::string>& fields,
  WriterB& piecesWriter,
  WriterI16& movesWriter
  ) {
  if (fields.size() != 22) {
    std::cerr << "Error: Expected 22 fields, got " << fields.size() << std::endl;
    throw std::invalid_argument("Invalid number of fields");
  }

  // <fen><score><move1.from><move1.to><move2.from><move2.to>...<move10.from><move10.to>

  // Parse position from FEN
  Position pos(fields[0]);
  
  // Create piece presence table: 768 booleans (12 channels × 64 squares)
  // Channels: piece_type (0-5) + color*6, where piece types are pawn=0, knight=1, etc.
  bool piecePresence[768];
  std::fill_n(piecePresence, 768, false);
  
  for (size_t i = 0; i < kNumSquares; ++i) {
    SafeSquare square = SafeSquare(i);
    ColoredPiece cp = pos.tiles_[square];
    
    if (cp != ColoredPiece::NO_COLORED_PIECE) {
      Piece pieceType = cp2p(cp);
      Color color = cp2color(cp);
      
      // Convert to target encoding: piece_type - 1 (pawn=0), color - 1 (white=0, black=1)
      int8_t encodedPiece = int8_t(pieceType - 1);  // 0-5
      int8_t colorIndex = int8_t(color - 1);        // 0-1
      
      // Channel index: piece_type + color*6
      size_t channel = encodedPiece + colorIndex * 6;
      size_t index = square + channel * 64;
      
      piecePresence[index] = true;
    }
  }
  piecesWriter.write_row(piecePresence);
  
  int16_t moveCoords[20];  // 10 moves × (from, to)
  for (size_t i = 2; i < fields.size(); ++i) {
    int sq = std::stoi(fields[i]);
    if (sq < 0 || sq > 64 * 6 + 1) {
      // Note: "sq == 64 * 6" is used for padding when there are fewer than 10 moves.
      std::cerr << "Error: Square index out of bounds: " << sq << std::endl;
      throw std::out_of_range("Square index out of bounds");
    }
    moveCoords[i - 2] = sq;
  }
  movesWriter.write_row(moveCoords);
}

int main(int argc, char *argv[]) {
  initialize_geometry();
  initialize_zorbrist();
  initialize_movegen();

  if (argc != 3 && argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input> <output> [limit]" << std::endl;
    return 1;
  }

  const std::string inpath = argv[1];
  const std::string outpath = argv[2];
  size_t limit = std::numeric_limits<size_t>::max();
  if (argc == 4) {
    try {
      limit = std::stoul(argv[3]);
    } catch (const std::exception& e) {
      std::cerr << "Invalid limit value: " << argv[3] << std::endl;
      return 1;
    }
  }

  std::ifstream infile(inpath);
  if (!infile.is_open()) {
    std::cerr << "Could not open input file: " << inpath << std::endl;
    return 1;
  }

  WriterB piecesWriter(outpath + "-pieces", { 768 });
  WriterI16 movesWriter(outpath + "-moves", { 20 });

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  size_t counter = 0;
  std::string line;
  while (std::getline(infile, line)) {
    if (line == "") {
      continue;
    }
    
    try {
      std::vector<std::string> parts = split(line, '|');
      process(parts, piecesWriter, movesWriter);
    } catch (const std::exception& e) {
      std::cerr << "Error processing line " << counter + 1 << ": " << e.what() << std::endl;
      std::cerr << "Line: " << line << std::endl;
      return 1;
    }

    if ((++counter) % 100'000 == 0) {
      double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
      std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
    }
    
    if (counter >= limit) {
      break;
    }
  }

  std::cout << "Completed " << counter << " positions." << std::endl;

  return 0;
}