
#ifndef SRC_EVAL_BYHAND_BYHAND_H
#define SRC_EVAL_BYHAND_BYHAND_H

#include <cstdint>
#include <memory>
#include <bit>
#include "../../game/Position.h"
#include "../../game/Utils.h"
#include "../../game/BoardListener.h"
#include "../../game/movegen/bishops.h"
#include "../Evaluator.h"
#include "../ColoredEvaluation.h"
#include "../pst/PieceSquareEvaluator.h"
#include "../nnue/Nnue.h"
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
  THEIR_PAWNxKNIGHT,
  THEIR_PAWNxBISHOP,
  THEIR_PAWNxROOK,
  THEIR_PAWNxQUEEN,
  THEIR_QUEENxKNIGHT,
  THEIR_QUEENxBISHOP,
  THEIR_QUEENxROOK,

  NUM_PAWN_TARGETS,
  NUM_KNIGHT_TARGETS,
  NUM_BISHOP_TARGETS,
  NUM_ROOK_TARGETS,
  NUM_QUEEN_TARGETS,

  NUM_PAWN_TARGETS_ON_THEIR_SIDE,
  NUM_KNIGHT_TARGETS_ON_THEIR_SIDE,
  NUM_BISHOP_TARGETS_ON_THEIR_SIDE,
  NUM_ROOK_TARGETS_ON_THEIR_SIDE,
  NUM_QUEEN_TARGETS_ON_THEIR_SIDE,

  NUM_PAWN_TARGETS_IN_CENTER,
  NUM_KNIGHT_TARGETS_IN_CENTER,
  NUM_BISHOP_TARGETS_IN_CENTER,
  NUM_ROOK_TARGETS_IN_CENTER,
  NUM_QUEEN_TARGETS_IN_CENTER,

  NUM_PAWNS_4th_RANK,
  NUM_PAWNS_5th_RANK,
  NUM_PAWNS_6th_RANK,
  NUM_KNIGHTS_ON_EDGE,

  NUM_SCARY_BISHOPS,
  NUM_SCARY_ROOKS,
  ROOKS_ON_OPEN_FILE,
  ROOKS_ON_SEMI_OPEN_FILE,
  CONNECTED_ROOKS,

  PINNED_MINORS,

  EF_COUNT
};

static constexpr Bitboard kWhiteRanks[8] = {
  kRanks[7],
  kRanks[6],
  kRanks[5],
  kRanks[4],
  kRanks[3],
  kRanks[2],
  kRanks[1],
  kRanks[0],
};

static constexpr Bitboard kBlackRanks[8] = {
  kRanks[0],
  kRanks[1],
  kRanks[2],
  kRanks[3],
  kRanks[4],
  kRanks[5],
  kRanks[6],
  kRanks[7],
};

template<Color US>
void pos2features(const Position& pos, const Threats& threats, int8_t *out) {
  static constexpr Color THEM = opposite_color<US>();
  static constexpr Direction FORWARD = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
  static constexpr Direction BACKWARD = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;
  static constexpr Direction FORWARD_EAST = US == Color::WHITE ? Direction::NORTH_EAST : Direction::SOUTH_EAST;
  static constexpr Direction BACKWARD_EAST = US == Color::WHITE ? Direction::SOUTH_EAST : Direction::NORTH_EAST;
  static constexpr Direction FORWARD_WEST = US == Color::WHITE ? Direction::NORTH_WEST : Direction::SOUTH_WEST;
  static constexpr Direction BACKWARD_WEST = US == Color::WHITE ? Direction::SOUTH_WEST : Direction::NORTH_WEST;

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

  const Bitboard ourRoyalty = ourQueens | ourKings;
  const Bitboard theirRoyalty = theirQueens | theirKings;
  const Bitboard ourHeavies = ourRoyalty | ourRooks;
  const Bitboard theirHeavies = theirRoyalty | theirRooks;
  const Bitboard ourMinors = ourKnights | ourBishops;
  const Bitboard theirMinors = theirKnights | theirBishops;

  static const Bitboard ourTargets = US == Color::WHITE ? threats.whiteTargets : threats.blackTargets;
  static const Bitboard theirTargets = US == Color::WHITE ? threats.blackTargets : threats.whiteTargets;

  const auto *ourRanks = US == Color::WHITE ? kWhiteRanks : kBlackRanks;
  const auto *theirRanks = US == Color::WHITE ? kBlackRanks : kWhiteRanks;
  const PawnAnalysis<US> pawnAnalysis(pos);

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

  out[EF::KING_ON_BACK_RANK] = std::popcount(ourRanks[0] & ourKings) - std::popcount(theirRanks[0] & theirKings);
  if (US == Color::WHITE) {
    out[EF::KING_ACTIVE] = (ourKingSq / 8 < 5) - (theirKingSq / 8 > 2);
  } else {
    out[EF::KING_ACTIVE] = (ourKingSq / 8 > 2) - (theirKingSq / 8 < 5);
  }

  out[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[2][theirKingSq] & ourTargets & ~theirTargets);
  out[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & theirTargets & ~ourTargets) - std::popcount(kNearby[3][theirKingSq] & ourTargets & ~theirTargets);

  out[EF::PASSED_PAWNS] = std::popcount(pawnAnalysis.ourPassedPawns) - std::popcount(pawnAnalysis.theirPassedPawns);
  out[EF::PASSED_PAWNS_7TH_RANK] = std::popcount(pawnAnalysis.ourPassedPawns & ourRanks[6]) - std::popcount(pawnAnalysis.theirPassedPawns & theirRanks[6]);
  out[EF::PASSED_PAWNS_6TH_RANK] = std::popcount(pawnAnalysis.ourPassedPawns & ourRanks[5]) - std::popcount(pawnAnalysis.theirPassedPawns & theirRanks[5]);
  out[EF::ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirIsolatedPawns);
  out[EF::DOUBLED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns) - std::popcount(pawnAnalysis.theirDoubledPawns);
  out[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns & pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirDoubledPawns & pawnAnalysis.theirIsolatedPawns);

  out[EF::HANGING_PAWN] = std::popcount(threats.badForOur<US>(Piece::PAWN) & ourPawns) - std::popcount(threats.badForOur<THEM>(Piece::PAWN) & theirPawns);
  out[EF::HANGING_KNIGHT] = std::popcount(threats.badForOur<US>(Piece::KNIGHT) & ourKnights) - std::popcount(threats.badForOur<THEM>(Piece::KNIGHT) & theirKnights);
  out[EF::HANGING_BISHOP] = std::popcount(threats.badForOur<US>(Piece::BISHOP) & ourBishops) - std::popcount(threats.badForOur<THEM>(Piece::BISHOP) & theirBishops);
  out[EF::HANGING_ROOK] = std::popcount(threats.badForOur<US>(Piece::ROOK) & ourRooks) - std::popcount(threats.badForOur<THEM>(Piece::ROOK) & theirRooks);
  out[EF::HANGING_QUEEN] = std::popcount(threats.badForOur<US>(Piece::QUEEN) & ourQueens) - std::popcount(threats.badForOur<THEM>(Piece::QUEEN) & theirQueens);

  out[EF::BISHOP_PAIR] = ((ourBishops & kWhiteSquares) && (ourBishops & kBlackSquares)) - ((theirBishops & kWhiteSquares) && (theirBishops & kBlackSquares));
  out[EF::PAWNxPAWN] = out[EF::OUR_PAWNS] * out[EF::OUR_PAWNS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_PAWNS];
  out[EF::PAWNxKNIGHT] = out[EF::OUR_PAWNS] * out[EF::OUR_KNIGHTS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_KNIGHTS];
  out[EF::PAWNxBISHOP] = out[EF::OUR_PAWNS] * out[EF::OUR_BISHOPS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_BISHOPS];
  out[EF::PAWNxROOK] = out[EF::OUR_PAWNS] * out[EF::OUR_ROOKS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_ROOKS];
  out[EF::PAWNxQUEEN] = out[EF::OUR_PAWNS] * out[EF::OUR_QUEENS] - out[EF::THEIR_PAWNS] * out[EF::THEIR_QUEENS];
  out[EF::QUEENxKNIGHT] = out[EF::OUR_QUEENS] * out[EF::OUR_KNIGHTS] - out[EF::THEIR_QUEENS] * out[EF::THEIR_KNIGHTS];
  out[EF::QUEENxBISHOP] = out[EF::OUR_QUEENS] * out[EF::OUR_BISHOPS] - out[EF::THEIR_QUEENS] * out[EF::THEIR_BISHOPS];
  out[EF::QUEENxROOK] = out[EF::OUR_QUEENS] * out[EF::OUR_ROOKS] - out[EF::THEIR_QUEENS] * out[EF::THEIR_ROOKS];

  out[EF::THEIR_PAWNxKNIGHT] = out[EF::THEIR_PAWNS] * out[EF::OUR_KNIGHTS] - out[EF::OUR_PAWNS] * out[EF::THEIR_KNIGHTS];
  out[EF::THEIR_PAWNxBISHOP] = out[EF::THEIR_PAWNS] * out[EF::OUR_BISHOPS] - out[EF::OUR_PAWNS] * out[EF::THEIR_BISHOPS];
  out[EF::THEIR_PAWNxROOK] = out[EF::THEIR_PAWNS] * out[EF::OUR_ROOKS] - out[EF::OUR_PAWNS] * out[EF::THEIR_ROOKS];
  out[EF::THEIR_PAWNxQUEEN] = out[EF::THEIR_PAWNS] * out[EF::OUR_QUEENS] - out[EF::OUR_PAWNS] * out[EF::THEIR_QUEENS];
  out[EF::THEIR_QUEENxKNIGHT] = out[EF::THEIR_QUEENS] * out[EF::OUR_KNIGHTS] - out[EF::OUR_QUEENS] * out[EF::THEIR_KNIGHTS];
  out[EF::THEIR_QUEENxBISHOP] = out[EF::THEIR_QUEENS] * out[EF::OUR_BISHOPS] - out[EF::OUR_QUEENS] * out[EF::THEIR_BISHOPS];
  out[EF::THEIR_QUEENxROOK] = out[EF::THEIR_QUEENS] * out[EF::OUR_ROOKS] - out[EF::OUR_QUEENS] * out[EF::THEIR_ROOKS];
  
  const Bitboard ourPawnTargets = US == Color::WHITE ? threats.whitePawnTargets : threats.blackPawnTargets;
  const Bitboard theirPawnTargets = US == Color::WHITE ? threats.blackPawnTargets : threats.whitePawnTargets;
  const Bitboard ourKnightTargets = US == Color::WHITE ? threats.whiteKnightTargets : threats.blackKnightTargets;
  const Bitboard theirKnightTargets = US == Color::WHITE ? threats.blackKnightTargets : threats.whiteKnightTargets;
  const Bitboard ourBishopTargets = US == Color::WHITE ? threats.whiteBishopTargets : threats.blackBishopTargets;
  const Bitboard theirBishopTargets = US == Color::WHITE ? threats.blackBishopTargets : threats.whiteBishopTargets;
  const Bitboard ourRookTargets = US == Color::WHITE ? threats.whiteRookTargets : threats.blackRookTargets;
  const Bitboard theirRookTargets = US == Color::WHITE ? threats.blackRookTargets : threats.whiteRookTargets;
  const Bitboard ourQueenTargets = US == Color::WHITE ? threats.whiteQueenTargets : threats.blackQueenTargets;
  const Bitboard theirQueenTargets = US == Color::WHITE ? threats.blackQueenTargets : threats.whiteQueenTargets;

  constexpr Bitboard theirSide = US == Color::WHITE ? (kRanks[0] | kRanks[1] | kRanks[2] | kRanks[3]) : (kRanks[7] | kRanks[6] | kRanks[5] | kRanks[4]);
  constexpr Bitboard ourSide = ~theirSide;

  out[EF::NUM_PAWN_TARGETS] = std::popcount(ourPawnTargets) - std::popcount(theirPawnTargets);
  out[EF::NUM_KNIGHT_TARGETS] = std::popcount(ourKnightTargets) - std::popcount(theirKnightTargets);
  out[EF::NUM_BISHOP_TARGETS] = std::popcount(ourBishopTargets) - std::popcount(theirBishopTargets);
  out[EF::NUM_ROOK_TARGETS] = std::popcount(ourRookTargets) - std::popcount(theirRookTargets);
  out[EF::NUM_QUEEN_TARGETS] = std::popcount(ourQueenTargets) - std::popcount(theirQueenTargets);

  out[EF::NUM_PAWN_TARGETS_ON_THEIR_SIDE] = std::popcount(ourPawnTargets & theirSide) - std::popcount(theirPawnTargets & ourSide);
  out[EF::NUM_KNIGHT_TARGETS_ON_THEIR_SIDE] = std::popcount(ourKnightTargets & theirSide) - std::popcount(theirKnightTargets & ourSide);
  out[EF::NUM_BISHOP_TARGETS_ON_THEIR_SIDE] = std::popcount(ourBishopTargets & theirSide) - std::popcount(theirBishopTargets & ourSide);
  out[EF::NUM_ROOK_TARGETS_ON_THEIR_SIDE] = std::popcount(ourRookTargets & theirSide) - std::popcount(theirRookTargets & ourSide);
  out[EF::NUM_QUEEN_TARGETS_ON_THEIR_SIDE] = std::popcount(ourQueenTargets & theirSide) - std::popcount(theirQueenTargets & ourSide);

  out[EF::NUM_PAWN_TARGETS_IN_CENTER] = std::popcount(ourPawnTargets & kCenter4) - std::popcount(theirPawnTargets & kCenter4);
  out[EF::NUM_KNIGHT_TARGETS_IN_CENTER] = std::popcount(ourKnightTargets & kCenter4) - std::popcount(theirKnightTargets & kCenter4);
  out[EF::NUM_BISHOP_TARGETS_IN_CENTER] = std::popcount(ourBishopTargets & kCenter4) - std::popcount(theirBishopTargets & kCenter4);
  out[EF::NUM_ROOK_TARGETS_IN_CENTER] = std::popcount(ourRookTargets & kCenter4) - std::popcount(theirRookTargets & kCenter4);
  out[EF::NUM_QUEEN_TARGETS_IN_CENTER] = std::popcount(ourQueenTargets & kCenter4) - std::popcount(theirQueenTargets & kCenter4);

  // Pawns that cannot move (forward or diagonally).
  const Bitboard ourBlockadedPawns = shift<BACKWARD>(theirPawns) & ourPawns & ~shift<BACKWARD_EAST>(theirPawns) & ~shift<BACKWARD_WEST>(theirPawns);
  const Bitboard theirBlockadedPawns = shift<FORWARD>(ourPawns) & theirPawns & ~shift<FORWARD_EAST>(ourPawns) & ~shift<FORWARD_WEST>(ourPawns);
  const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(ourBishops, ourBlockadedPawns);
  const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(theirBishops, theirBlockadedPawns);

  static const Bitboard edges = kRanks[0] | kRanks[7] | kFiles[FILE_A] | kFiles[FILE_H];

  out[EF::NUM_PAWNS_4th_RANK] = std::popcount(ourPawns & ourRanks[3]) - std::popcount(theirPawns & theirRanks[3]);
  out[EF::NUM_PAWNS_5th_RANK] = std::popcount(ourPawns & ourRanks[4]) - std::popcount(theirPawns & theirRanks[4]);
  out[EF::NUM_PAWNS_6th_RANK] = std::popcount(ourPawns & ourRanks[5]) - std::popcount(theirPawns & theirRanks[5]);
  out[EF::NUM_KNIGHTS_ON_EDGE] = std::popcount(ourKnights & edges) - std::popcount(theirKnights & edges);

  out[EF::NUM_SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirHeavies) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourHeavies);
  out[EF::NUM_SCARY_ROOKS] = std::popcount(ourRookTargets & theirRoyalty) - std::popcount(theirRookTargets & ourRoyalty);

  const Bitboard openFiles = pawnAnalysis.filesWithoutOurPawns & pawnAnalysis.filesWithoutTheirPawns;
  out[EF::ROOKS_ON_OPEN_FILE] = std::popcount(ourRooks & openFiles) - std::popcount(theirRooks & openFiles);
  out[EF::ROOKS_ON_SEMI_OPEN_FILE] = std::popcount(ourRooks & pawnAnalysis.filesWithoutOurPawns & ~pawnAnalysis.filesWithoutTheirPawns) - std::popcount(theirRooks & pawnAnalysis.filesWithoutTheirPawns & ~pawnAnalysis.filesWithoutOurPawns);

  UnsafeSquare ourRookSq = lsb_or_none(ourRooks);
  UnsafeSquare theirRookSq = lsb_or_none(theirRooks);
  // Note: "kFiles[ourRookSq % 8]" will return kFiles[0] if ourRookSq == NO_SQUARE, but this is fine, since "& ourRooks" will always return zero.
  out[EF::CONNECTED_ROOKS] = (std::popcount((kFiles[File(ourRookSq % 8)] | square2rank(ourRookSq)) & ourRooks) >= 2) - (std::popcount((kFiles[File(theirRookSq % 8)] | square2rank(theirRookSq)) & theirRooks) >= 2);

  PinMasks ourPinnedMask = compute_pin_masks<US>(pos, ourKingSq);
  PinMasks theirPinnedMask = compute_pin_masks<THEM>(pos, theirKingSq);
  out[EF::PINNED_MINORS] = 
    std::popcount((ourPinnedMask.all & ourKnights) | ((ourPinnedMask.horizontal | ourPinnedMask.vertical) & ourBishops))
    -
    std::popcount((theirPinnedMask.all & theirKnights) | ((theirPinnedMask.horizontal | theirPinnedMask.vertical) & theirBishops));
}

struct ByHandEvaluator : public EvaluatorInterface {
  NNUE::Matrix<2, EF::EF_COUNT, int16_t> weights;
  NNUE::Vector<2, int16_t> bias;
  NNUE::Vector<EF::EF_COUNT, int8_t> x;
  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::WHITE> alpha, ColoredEvaluation<Color::WHITE> beta) override {
    return _evaluate<Color::WHITE>(pos, threats, alpha, beta);
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::BLACK> alpha, ColoredEvaluation<Color::BLACK> beta) override {
    return _evaluate<Color::BLACK>(pos, threats, alpha, beta);
  }

  template<Color US>
  ColoredEvaluation<US> _evaluate(const Position& pos, const Threats& threats, ColoredEvaluation<US> alpha, ColoredEvaluation<US> beta) {
    pos2features<US>(pos, threats, x.data_ptr());
    int32_t late = bias[1];
    int32_t early = bias[0];
    for (size_t i = 0; i < EF::EF_COUNT; ++i) {
      late += x[i] * weights(0, i);
      early += x[i] * weights(1, i);
    }
    int32_t earliness = 0;
    earliness += x[EF::OUR_KNIGHTS] + x[EF::OUR_BISHOPS] + x[EF::OUR_ROOKS] + x[EF::OUR_QUEENS] * 3;
    earliness += x[EF::THEIR_KNIGHTS] + x[EF::THEIR_BISHOPS] + x[EF::THEIR_ROOKS] + x[EF::THEIR_QUEENS] * 3;
    earliness = std::min(18, earliness);
    int32_t r = (early * earliness + late * (18 - earliness)) / 18;
    return ColoredEvaluation<US>(r);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    auto copy = std::make_shared<ByHandEvaluator>();
    copy->weights = this->weights;
    copy->bias = this->bias;
    return copy;
  }

  void load_from_stream(std::istream& in) {
    weights.load_from_stream(in, "weights");
    bias.load_from_stream(in, "bias");
  }

  std::string to_string() const override {
    return "ByHandEvaluator";
  }

  void place_piece(ColoredPiece cp, SafeSquare square) override {}
  void remove_piece(ColoredPiece cp, SafeSquare square) override {}
  void place_piece(SafeColoredPiece cp, SafeSquare square) override {}
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {}
  void empty() override {}
};

};  // namespace ByHand

}  // namespace ChessEngine

#endif // SRC_EVAL_BYHAND_BYHAND_H
