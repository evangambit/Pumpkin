#ifndef SRC_EVAL_BYHAND_BYHAND_H
#define SRC_EVAL_BYHAND_BYHAND_H

#include <memory>
#include <bit>
#include "../../game/Position.h"
#include "../../game/Utils.h"
#include "../../game/BoardListener.h"
#include "../Evaluator.h"
#include "../ColoredEvaluation.h"
#include "../pst/PieceSquareEvaluator.h"
#include "../PawnAnalysis.h"

namespace ChessEngine {

namespace ByHand {

enum EF {
  OUR_PAWNS,
  OUR_KNIGHTS,
  OUR_BISHOPS,
  OUR_ROOKS,
  OUR_QUEENS,
  THEIR_PAWNS,
  THEIR_KNIGHTS,
  THEIR_BISHOPS,
  THEIR_ROOKS,
  THEIR_QUEENS,

  KING_ON_BACK_RANK,
  KING_ON_CENTER_FILE,
  KING_ACTIVE,
  THREATS_NEAR_KING_2,
  THREATS_NEAR_KING_3,
  PASSED_PAWNS,
  PASSED_PAWNS_7TH_RANK,
  PASSED_PAWNS_6TH_RANK,

  ISOLATED_PAWNS,
  DOUBLED_PAWNS,
  DOUBLE_ISOLATED_PAWNS,

  HANGING_PAWN,
  HANGING_KNIGHT,
  HANGING_BISHOP,
  HANGING_ROOK,
  HANGING_QUEEN,

  BISHOP_PAIR,

  // Interaction terms with our pawn and queen.
  PAWNxPAWN,
  PAWNxKNIGHT,
  PAWNxBISHOP,
  PAWNxROOK,
  PAWNxQUEEN,
  QUEENxKNIGHT,
  QUEENxBISHOP,
  QUEENxROOK,

  // Interaction terms with opponent's pawn and queen.
  THEIR_PAWNxPAWN,
  THEIR_PAWNxKNIGHT,
  THEIR_PAWNxBISHOP,
  THEIR_PAWNxROOK,
  THEIR_PAWNxQUEEN,
  THEIR_QUEENxKNIGHT,
  THEIR_QUEENxBISHOP,
  THEIR_QUEENxROOK,

  EF_COUNT
};

template<Color US>
void pos2features(const Position& pos, const Threats& threats, int8_t *out) {
  constexpr Color THEM = opposite_color<US>();

  assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
  assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);
  const SafeSquare ourKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
  const SafeSquare theirKingSq = lsb_i_promise_board_is_not_empty(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

  const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
  const Bitboard ourKnights = pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()];
  const Bitboard ourBishops = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()];
  const Bitboard ourRooks = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()];
  const Bitboard ourQueens = pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
  const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];

  const Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
  const Bitboard theirKnights = pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()];
  const Bitboard theirBishops = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()];
  const Bitboard theirRooks = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()];
  const Bitboard theirQueens = pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
  const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

  PawnAnalysis<US> pawnAnalysis(pos);

  out[EF::OUR_PAWNS] = std::popcount(ourPawns);
  out[EF::OUR_KNIGHTS] = std::popcount(ourKnights);
  out[EF::OUR_BISHOPS] = std::popcount(ourBishops);
  out[EF::OUR_ROOKS] = std::popcount(ourRooks);
  out[EF::OUR_QUEENS] = std::popcount(ourQueens);
  out[EF::THEIR_PAWNS] = std::popcount(theirPawns);
  out[EF::THEIR_KNIGHTS] = std::popcount(theirKnights);
  out[EF::THEIR_BISHOPS] = std::popcount(theirBishops);
  out[EF::THEIR_ROOKS] = std::popcount(theirRooks);
  out[EF::THEIR_QUEENS] = std::popcount(theirQueens);

  if (US == Color::WHITE) {
    out[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 7) - (theirKingSq / 8 == 0);
    out[EF::KING_ACTIVE] = (ourKingSq / 8 < 5) - (theirKingSq / 8 > 2);
  } else {
    out[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 0) - (theirKingSq / 8 == 7);
    out[EF::KING_ACTIVE] = (ourKingSq / 8 > 2) - (theirKingSq / 8 < 5);
  }

  const Bitboard ourTargets = US == Color::WHITE ? threats.whiteTargets : threats.blackTargets;
  const Bitboard theirTargets = US == Color::WHITE ? threats.blackTargets : threats.whiteTargets;
  const Bitboard our7thRank = US == Color::WHITE ? kRanks[1] : kRanks[6];
  const Bitboard their7thRank = US == Color::WHITE ? kRanks[6] : kRanks[1];
  const Bitboard our6thRank = US == Color::WHITE ? kRanks[2] : kRanks[5];
  const Bitboard their6thRank = US == Color::WHITE ? kRanks[5] : kRanks[2];

  out[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[2][theirKingSq] & ourTargets & ~theirTargets);
  out[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[3][theirKingSq] & ourTargets & ~theirTargets);

  out[EF::PASSED_PAWNS] = std::popcount(pawnAnalysis.ourPassedPawns) - std::popcount(pawnAnalysis.theirPassedPawns);
  out[EF::PASSED_PAWNS_7TH_RANK] = std::popcount(pawnAnalysis.ourPassedPawns & our7thRank) - std::popcount(pawnAnalysis.theirPassedPawns & their7thRank);
  out[EF::PASSED_PAWNS_6TH_RANK] = std::popcount(pawnAnalysis.ourPassedPawns & our6thRank) - std::popcount(pawnAnalysis.theirPassedPawns & their6thRank);
  out[EF::ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirIsolatedPawns);
  out[EF::DOUBLED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns) - std::popcount(pawnAnalysis.theirDoubledPawns);
  out[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns & pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirDoubledPawns & pawnAnalysis.theirIsolatedPawns);

  out[EF::HANGING_PAWN] = std::popcount(threats.badForOur<US>(Piece::PAWN) & ourPawns) - std::popcount(threats.badForOur<THEM>(Piece::PAWN) & theirPawns);
  out[EF::HANGING_KNIGHT] = std::popcount(threats.badForOur<US>(Piece::KNIGHT) & ourKnights) - std::popcount(threats.badForOur<THEM>(Piece::KNIGHT) & theirKnights);
  out[EF::HANGING_BISHOP] = std::popcount(threats.badForOur<US>(Piece::BISHOP) & ourBishops) - std::popcount(threats.badForOur<THEM>(Piece::BISHOP) & theirBishops);
  out[EF::HANGING_ROOK] = std::popcount(threats.badForOur<US>(Piece::ROOK) & ourRooks) - std::popcount(threats.badForOur<THEM>(Piece::ROOK) & theirRooks);
  out[EF::HANGING_QUEEN] = std::popcount(threats.badForOur<US>(Piece::QUEEN) & ourQueens) - std::popcount(threats.badForOur<THEM>(Piece::QUEEN) & theirQueens);

  out[EF::PAWNxPAWN] = out[EF::OUR_PAWNS] * out[EF::OUR_PAWNS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_PAWNS];
  out[EF::PAWNxKNIGHT] = out[EF::OUR_PAWNS] * out[EF::OUR_KNIGHTS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_KNIGHTS];
  out[EF::PAWNxBISHOP] = out[EF::OUR_PAWNS] * out[EF::OUR_BISHOPS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_BISHOPS];
  out[EF::PAWNxROOK] = out[EF::OUR_PAWNS] * out[EF::OUR_ROOKS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_ROOKS];
  out[EF::PAWNxQUEEN] = out[EF::OUR_PAWNS] * out[EF::OUR_QUEENS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_QUEENS];
  out[EF::QUEENxKNIGHT] = out[EF::OUR_QUEENS] * out[EF::OUR_KNIGHTS] - out[EF::THEIR_QUEENS] * out[EF::THEIR_KNIGHTS];
  out[EF::QUEENxBISHOP] = out[EF::OUR_QUEENS] * out[EF::OUR_BISHOPS] - out[EF::THEIR_QUEENS] * out[EF::THEIR_BISHOPS];
  out[EF::QUEENxROOK] = out[EF::OUR_QUEENS] * out[EF::OUR_ROOKS] - out[EF::THEIR_QUEENS] * out[EF::THEIR_ROOKS];

  out[EF::THEIR_PAWNxPAWN] = out[EF::THEIR_PAWNS] * out[EF::OUR_PAWNS] - out[EF::OUR_PAWNS] * out[EF::THEIR_PAWNS];
  out[EF::THEIR_PAWNxKNIGHT] = out[EF::THEIR_PAWNS] * out[EF::OUR_KNIGHTS] - out[EF::OUR_PAWNS] * out[EF::THEIR_KNIGHTS];
  out[EF::THEIR_PAWNxBISHOP] = out[EF::THEIR_PAWNS] * out[EF::OUR_BISHOPS] - out[EF::OUR_PAWNS] * out[EF::THEIR_BISHOPS];
  out[EF::THEIR_PAWNxROOK] = out[EF::THEIR_PAWNS] * out[EF::OUR_ROOKS] - out[EF::OUR_PAWNS] * out[EF::THEIR_ROOKS];
  out[EF::THEIR_PAWNxQUEEN] = out[EF::THEIR_PAWNS] * out[EF::OUR_QUEENS] - out[EF::OUR_PAWNS] * out[EF::THEIR_QUEENS];
  out[EF::THEIR_QUEENxKNIGHT] = out[EF::THEIR_QUEENS] * out[EF::OUR_KNIGHTS] - out[EF::OUR_QUEENS] * out[EF::THEIR_KNIGHTS];
  out[EF::THEIR_QUEENxBISHOP] = out[EF::THEIR_QUEENS] * out[EF::OUR_BISHOPS] - out[EF::OUR_QUEENS] * out[EF::THEIR_BISHOPS];
  out[EF::THEIR_QUEENxROOK] = out[EF::THEIR_QUEENS] * out[EF::OUR_ROOKS] - out[EF::OUR_QUEENS] * out[EF::THEIR_ROOKS];
  
}

struct ByHandEvaluator : public PieceSquareEvaluator {
  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::WHITE> alpha, ColoredEvaluation<Color::WHITE> beta) override {
    ColoredEvaluation<Color::WHITE> pst = PieceSquareEvaluator::evaluate_white(pos, threats, plyFromRoot, alpha, beta);
    return pst;
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<ByHandEvaluator>();
  }

  std::string to_string() const override {
    return "ByHandEvaluator";
  }
};

};  // namespace ByHand

}  // namespace ChessEngine

#endif // SRC_EVAL_BYHAND_BYHAND_H
