#include "../game/Geometry.h"
#include "../game/Utils.h"
#include "../game/Position.h"
#include "../game/movegen/movegen.h"
#include "../game/movegen/sliding.h"
#include "nnue/ShardedMatrix.h"
#include "nnue/Nnue.h"
#include "nnue/NnueEvaluator.h"
#include "qst/QstEvaluator.h"

#include <gflags/gflags.h>
#include <memory>
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
DEFINE_bool(emit_piece_counts, false, "Whether to emit piece counts");

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

struct EvaluatedData {
  bool qstFeatures[Q_NUM_FEATURES * 64];
  NNUE::Features nnueFeatures;
  int16_t wdl[3];
  int8_t pieceCounts[10];
  bool valid = true;
};

struct ProcessContext {
  QstEvaluator qstEvaluator;
  std::shared_ptr<NNUE::Nnue<int16_t>> nnue;
  std::unique_ptr<NNUE::NnueEvaluator<int16_t>> nnueEvaluator;
  
  ProcessContext() {
    nnue = std::make_shared<NNUE::Nnue<int16_t>>();
    nnueEvaluator = std::make_unique<NNUE::NnueEvaluator<int16_t>>(nnue);
  }
};

EvaluatedData process_line(
  const std::vector<std::string>& line,
  ProcessContext *ctx
  ) {
  EvaluatedData data;
  if (line.size() != 4 && line.size() != 2) {
    std::cout << "error: line has " << line.size() << " elements" << std::endl;
    data.valid = false;
    return data;
  }

  // If line has 2 elements, it's in the format: fen | score
  // If line has 4 elements, it's in the format: fen | win | draw | loss

  Position pos(line[0]);
  Threats threats;
  create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);

  if (FLAGS_emit_qst) {
    std::vector<Bitboard> features;
    features.reserve(Q_NUM_FEATURES);
    if (pos.turn_ == Color::WHITE) {
      ctx->qstEvaluator.get_features<Color::WHITE>(pos, threats, &features);
    } else {
      ctx->qstEvaluator.get_features<Color::BLACK>(pos, threats, &features);
    }
    
    if (features.size() != Q_NUM_FEATURES) {
      std::cerr << "Error: Expected " << Q_NUM_FEATURES << " features, got " << features.size() << std::endl;
      data.valid = false;
      return data;
    }

    for (size_t i = 0; i < Q_NUM_FEATURES; ++i) {
      for (size_t j = 0; j < 64; ++j) {
        data.qstFeatures[i * 64 + j] = (features[i] >> j) & 1;
      }
    }
  }

  if (FLAGS_emit_nnue) {
    data.nnueFeatures = NNUE::pos2features(ctx->nnueEvaluator.get(), pos, threats);
    if (pos.turn_ == Color::BLACK) {
      data.nnueFeatures.flip_();
    }
  }

  if (line.size() == 4) {
    data.wdl[0] = int16_t(std::stoi(line[1]));
    data.wdl[1] = int16_t(std::stoi(line[2]));
    data.wdl[2] = int16_t(std::stoi(line[3]));
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
    float winRate = NNUE::sigmoid(1.25f * score - 1.33f);
    float loseRate = NNUE::sigmoid(-1.25f * score - 1.33f);
    data.wdl[0] = int16_t(std::round(winRate * 1000.0f));
    data.wdl[2] = int16_t(std::round(loseRate * 1000.0f));
    data.wdl[1] = 1000 - data.wdl[0] - data.wdl[2];
  }

  if (FLAGS_emit_piece_counts) {
    data.pieceCounts[0] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]);
    data.pieceCounts[1] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT]);
    data.pieceCounts[2] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
    data.pieceCounts[3] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK]);
    data.pieceCounts[4] = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
    data.pieceCounts[5] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]);
    data.pieceCounts[6] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT]);
    data.pieceCounts[7] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
    data.pieceCounts[8] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK]);
    data.pieceCounts[9] = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);
  }
  
  return data;
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
  
  std::ofstream sparseNnueValueWriter(outpath + "-nnue-sparse-values", std::ios::binary);
  std::ofstream sparseNnueLengthWriter(outpath + "-nnue-sparse-lengths", std::ios::binary);

  std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

  const unsigned numThreads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
  std::vector<ProcessContext> contexts(numThreads);

  size_t counter = 0;
  std::string line;
  
  constexpr size_t CHUNK_SIZE = 16384;
  std::vector<std::string> lines;
  lines.reserve(CHUNK_SIZE);

  auto process_chunk = [&]() {
    if (lines.empty()) return;

    std::vector<EvaluatedData> results(lines.size());
    std::vector<std::thread> workers;
    
    // Divide work
    for (unsigned t = 0; t < numThreads; ++t) {
      workers.emplace_back([t, numThreads, &lines, &results, &contexts]() {
        for (size_t i = t; i < lines.size(); i += numThreads) {
          std::vector<std::string> parts = split(lines[i], '|');
          results[i] = process_line(parts, &contexts[t]);
        }
      });
    }

    for (auto& w : workers) {
      w.join();
    }

    // Write results sequentially
    for (size_t i = 0; i < results.size(); ++i) {
      if (!results[i].valid) continue;
      
      if (FLAGS_emit_qst) {
        qstInputWriter.write_row(results[i].qstFeatures);
      }
      if (FLAGS_emit_nnue) {
        sparseNnueValueWriter.write(reinterpret_cast<const char*>(results[i].nnueFeatures.onIndices), sizeof(results[i].nnueFeatures.onIndices[0]) * results[i].nnueFeatures.length);
        sparseNnueLengthWriter.write(reinterpret_cast<const char*>(&results[i].nnueFeatures.length), sizeof(results[i].nnueFeatures.length));
      }
      
      evalWriter.write_row(results[i].wdl);
      if (FLAGS_emit_piece_counts) {
        pieceCountWriter.write_row(results[i].pieceCounts);
      }
      
      if ((++counter) % 100'000 == 0) {
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
        std::cout << "Finished " << counter / 1000 << "k in " << ms / 1000 << " seconds" << std::endl;
      }
    }
    lines.clear();
  };

  while (std::getline(std::cin, line)) {
    if (line == "") {
      continue;
    }
    lines.push_back(line);
    if (lines.size() >= CHUNK_SIZE) {
      process_chunk();
    }
  }
  process_chunk();

  sparseNnueValueWriter.close();
  sparseNnueLengthWriter.close();

  std::cout << "Completed " << counter << " positions." << std::endl;

  return 0;
}
