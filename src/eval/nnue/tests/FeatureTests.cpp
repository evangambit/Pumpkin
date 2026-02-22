#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>

#include "../Nnue.h"
#include "../NnueEvaluator.h"
#include "../../../game/Position.h"
#include "../../../game/movegen/movegen.h"

using namespace NNUE;
using namespace ChessEngine;

/**
 * This file is to ensure pos2features and NnueEvaluator are aligned.
 */
class NNUETest : public ::testing::Test {
 protected:
  void SetUp() override {
    initialize_zorbrist();
    initialize_geometry();
    initialize_movegen();
  }
};

void test_fen(const std::string& fen, std::shared_ptr<Nnue<int16_t>> nnue) {
  Position pos(fen);
  NnueEvaluator evaluator(nnue);
  Threats threats;
  create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);

  Evaluation eval = evaluator._evaluate(pos, threats);

  std::vector<int16_t> features;
  for (size_t i = 0; i < NNUE_INPUT_DIM; ++i) {
    if (nnue->x[i]) {
      features.push_back(i);
    }
  }

  EXPECT_EQ(eval, evaluator.from_scratch(pos, threats)) << "Evaluation mismatch for position: " << fen;

  std::vector<uint16_t> features2 = pos2features(&evaluator, pos, threats).to_vector();

  std::sort(features.begin(), features.end());
  std::sort(features2.begin(), features2.end());

  EXPECT_EQ(features.size(), features2.size());
  for (size_t i = 0; i < features.size(); ++i) {
    EXPECT_EQ(features[i], features2[i]) << "Feature index: " << i;
  }
}

TEST_F(NNUETest, TestLotsOfFens) {
  std::vector<std::string> fens = {
    "4k3/p1R4p/6p1/4Kp2/2P4P/P1Pr1PP1/8/8 w - - 11 42",
    "r1bq1rk1/p2nbppp/p1p1p3/3pP3/3P4/1PN2N2/P1P2PPP/R1BQ1RK1 b - - 0 9",
    "rnbqkb1r/ppp2ppp/3p4/3p2P1/3P4/8/PPP1PP1P/R1BQKBNR w KQkq - 0 4",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2BP4/4P3/PPP2PPP/RNBQK1NR w KQkq - 0 2",
    "2b1rr1k/3nppb1/p2p2p1/N2P3p/N6P/5B2/1PP1RPP1/3R2K1 w - - 1 24",
    "8/8/R7/8/pk5P/5N2/3r4/7K b - - 2 65",
    "r1bq1rk1/pp4pp/5p2/2Pp1nB1/Q3n2P/P4NP1/1P2PP2/R3KB1R w KQ - 0 14",
    "6R1/8/5k2/4b3/7P/3r2PK/8/8 w - - 22 77",
    "6k1/p1p2p2/2pp4/4P2P/1q4b1/3QN1b1/P1P1R1P1/6K1 w - - 2 32",
    "r2qr1k1/1b5p/p2b2pP/P2p1p2/1p1Nn3/1P1BP1P1/2P1NP2/R2Q1K1R b - - 0 20",
    "3rq1k1/2pn1pp1/8/2pPpN1p/4P3/pP3PP1/P1Q3KP/5B2 w - - 0 28",
    "8/5K2/r7/8/4kn2/8/8/3R4 b - - 84 92",
    "rnbqkbnr/p1p1pppp/3p4/1p6/6P1/7B/PPPPPP1P/RNBQK1NR w KQkq - 0 1",
    "3r2k1/3r1pp1/4n2p/3pP2P/p5P1/P1BRK3/1P1R4/8 w - - 23 65",
    "5k2/2p2n2/1p1bB3/p6P/P5P1/2P5/1P3PK1/8 b - - 0 41",
    "r1bqk2r/pp1n1ppp/3b1n2/3p4/3p4/P4NP1/1PQNPPBP/R1B2RK1 b kq - 3 10",
    "r1bq1rk1/ppp2ppp/2n2n2/3p4/2PP1N2/P1P1B3/5PPP/R2QKB1R b KQ - 0 10",
    "5r2/1kp1r3/1pn3pp/3nB3/p2PR2P/P7/1P1N1PP1/4RK2 w - - 2 27",
    "8/8/4k1p1/R5P1/2P5/2K5/7r/8 w - - 2 55",
    "3r2k1/1q1r1pp1/1p2p2p/p1bn4/P2N4/2P4P/1PQ2PP1/2BRR2K b - - 3 31",
    "r2q1rk1/pb3pp1/1pn1p2p/3n4/P1NP4/1P1B4/2NQ1PPP/R3R1K1 w - - 2 18",
    "5rk1/Q5p1/3q3p/1p2p3/7P/1P2n3/P1P2PbR/R5K1 b - - 1 23",
    "r1bq1rk1/3p1ppp/p1n1pn2/b7/2PP4/P4N2/2Q1BPPP/RNB2RK1 w - - 3 10",
    "8/8/2p1k3/1p2B3/pPbKP3/P1P5/8/8 w - - 37 79",
    "2br4/p3r1k1/1p1qnppp/PPpBp3/4P3/3P1NP1/4QP1P/RR4K1 w - - 9 26",
    "q5k1/1b4p1/1p1p1p1p/1Pp1pP1P/2PnP1P1/2BP2K1/8/2QB4 b - - 52 66",
    "rnbqk1nr/p3ppb1/2pp2p1/1p5p/3PP3/2N2NQ1/PPP2PPP/R1B1KB1R w KQkq - 4 6",
    "r7/2R1bk1p/6pP/2N5/3p1P2/P7/1P6/6K1 w - - 3 44",
    "2r1r2k/3q2pp/Pp1npp2/1Pbp4/5P2/3BP1P1/1QR3KP/R2N4 w - - 4 29",
    "r3k2r/pp1nnppp/2pp1q2/2b1p3/P3P1b1/1PNP1N2/2P1BPPP/R1BQ1RK1 b kq - 0 7",
    "r1b1kb1r/3n2p1/p1p3q1/7p/2P1PP2/2N2R1P/P3Q1P1/R1B3K1 b - - 0 24",
    "8/1b6/8/3p4/7B/4Kp2/6k1/8 w - - 68 117",
    "1n6/1k6/p1q5/8/2p1N3/P7/1PP2Q2/2K5 w - - 0 48",
    "r2q1rk1/1p1n1ppp/2p1pb2/p2p4/3P1BQ1/P3PN2/RPP2PPP/3R2K1 b - - 3 12",
    "rnbqkb1r/ppp1p1pp/5n2/3p1p2/1P6/5NP1/P1PPPPBP/RNBQK2R b KQkq - 3 2",
    "r1bq1rk1/p1p3pp/1n1pPb2/2p5/8/1PN2NP1/PBPPQ2P/2KR3R w - - 1 14",
    "8/8/6k1/7p/6p1/8/4K2P/8 b - - 0 54",
    "r2qkb1r/1p1n1pp1/2p1pn1p/p2p1b2/2P2P2/NP2PN2/PB1PB1PP/R2Q1RK1 b kq - 3 7",
    "8/1p6/8/p1p5/P5rk/1PPP1K1p/7R/8 w - - 3 66",
    "r2qkbnr/1pp1pp1p/p1n5/3p2P1/3P4/N4N1R/PPP1PP2/R1BQK3 b Qkq - 0 7",
    "7k/3N3p/rpp1r1p1/4N1n1/5p2/P2RnP2/6PP/2R4K b - - 5 33",
    "3r4/4bpk1/p1p1q1p1/2p1p2p/PrN1P2P/1PQP1RP1/5PK1/2R5 b - - 43 56",
    "2b4r/2P1k3/5p2/4p2P/p7/2R5/n2NB3/4K3 w - - 0 50",
    "b4rk1/5pb1/2p1pnpn/2PpN2p/p2P1P2/3BPN1P/3B1KP1/1R6 b - - 4 23",
    "6r1/r6k/1Rp1pp2/n2p1p1p/P2P3P/2Q1PqPN/5P1K/6R1 b - - 3 49",
    "5k2/3K4/4Pb1p/pB6/Pp4P1/1P6/8/8 w - - 11 115",
    "8/1r1rk1p1/2R2p2/R2p2p1/3P4/4P1KP/5P2/8 w - - 63 68",
    "6k1/3nbpp1/4p3/2Pqn1P1/p3N3/P1P1B2p/2B4P/K5Q1 w - - 5 38",
    "1r1q1rk1/5ppp/8/1ppBp3/3nP1bb/1P1P4/1BPQ2P1/1K3RNR b - - 1 16",
    "8/5pB1/2b1k1p1/4P3/4PP2/4K3/8/8 b - - 29 59",
  };

  std::shared_ptr<Nnue<int16_t>> nnue = std::make_shared<Nnue<int16_t>>();
  nnue->randn_();
  for (const auto& fen : fens) {
    test_fen(fen, nnue);
  }
}

TEST_F(NNUETest, Foo) {
  Position pos("r1b1kb1r/ppqppppp/2n5/8/4n3/8/PPPP1PPP/R1BQKBNR w KQkq - 0 5");
  Threats threats;
  create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
  auto nnue = std::make_shared<Nnue<int16_t>>();
  NnueEvaluator<int16_t> evaluator(nnue);
  auto features = pos2features(&evaluator, pos, threats).to_vector();
  std::sort(features.begin(), features.end());
  for (int i = 0; i < features.size(); ++i) {
    std::cout << features[i] / 64 << " " << features[i] % 64 << std::endl;
  }
  std::cout << std::endl;
}