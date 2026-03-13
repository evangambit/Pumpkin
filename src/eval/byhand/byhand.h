
#ifndef SRC_EVAL_BYHAND_BYHAND_H
#define SRC_EVAL_BYHAND_BYHAND_H

#include <cstdint>
#include <memory>
#include <bit>
#include "../../game/Position.h"
#include "../../game/Utils.h"
#include "../../game/BoardListener.h"
#include "../../game/Geometry.h"
#include "../../game/movegen/bishops.h"
#include "../Evaluator.h"
#include "../ColoredEvaluation.h"
#include "../pst/PieceSquareEvaluator.h"
#include "../nnue/Nnue.h"
#include "../PawnAnalysis.h"

namespace ChessEngine {

namespace ByHand {

enum EF {
  EARLINESS = 0,

  PAWNS,
  KNIGHTS,
  BISHOPS,
  ROOKS,
  QUEENS,

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
  BACK_RANK_MATE_THREAT,
  KING_TROPISM,
  OPPOSITION,
  ONLY_HAVE_PAWNS,
  LONELY_KING_DIST_TO_EDGE,
  LONELY_KING_DIST_TO_CORNER,
  LONELY_KING_DIST_TO_KING,

  EF_COUNT
};

inline std::string to_string(EF e) {
  switch (e) {
    case EARLINESS: return "EARLINESS";
    case PAWNS: return "PAWNS";
    case KNIGHTS: return "KNIGHTS";
    case BISHOPS: return "BISHOPS";
    case ROOKS: return "ROOKS";
    case QUEENS: return "QUEENS";
    case KING_ON_BACK_RANK: return "KING_ON_BACK_RANK";
    case KING_ACTIVE: return "KING_ACTIVE";
    case THREATS_NEAR_KING_2: return "THREATS_NEAR_KING_2";
    case THREATS_NEAR_KING_3: return "THREATS_NEAR_KING_3";
    case PASSED_PAWNS: return "PASSED_PAWNS";
    case PASSED_PAWNS_7TH_RANK: return "PASSED_PAWNS_7TH_RANK";
    case PASSED_PAWNS_6TH_RANK: return "PASSED_PAWNS_6TH_RANK";
    case ISOLATED_PAWNS: return "ISOLATED_PAWNS";
    case DOUBLED_PAWNS: return "DOUBLED_PAWNS";
    case DOUBLE_ISOLATED_PAWNS: return "DOUBLE_ISOLATED_PAWNS";
    case HANGING_PAWN: return "HANGING_PAWN";
    case HANGING_KNIGHT: return "HANGING_KNIGHT";
    case HANGING_BISHOP: return "HANGING_BISHOP";
    case HANGING_ROOK: return "HANGING_ROOK";
    case HANGING_QUEEN: return "HANGING_QUEEN";
    case BISHOP_PAIR: return "BISHOP_PAIR";
    case NUM_PAWN_TARGETS: return "NUM_PAWN_TARGETS";
    case NUM_KNIGHT_TARGETS: return "NUM_KNIGHT_TARGETS";
    case NUM_BISHOP_TARGETS: return "NUM_BISHOP_TARGETS";
    case NUM_ROOK_TARGETS: return "NUM_ROOK_TARGETS";
    case NUM_QUEEN_TARGETS: return "NUM_QUEEN_TARGETS";
    case NUM_PAWN_TARGETS_ON_THEIR_SIDE: return "NUM_PAWN_TARGETS_ON_THEIR_SIDE";
    case NUM_KNIGHT_TARGETS_ON_THEIR_SIDE: return "NUM_KNIGHT_TARGETS_ON_THEIR_SIDE";
    case NUM_BISHOP_TARGETS_ON_THEIR_SIDE: return "NUM_BISHOP_TARGETS_ON_THEIR_SIDE";
    case NUM_ROOK_TARGETS_ON_THEIR_SIDE: return "NUM_ROOK_TARGETS_ON_THEIR_SIDE";
    case NUM_QUEEN_TARGETS_ON_THEIR_SIDE: return "NUM_QUEEN_TARGETS_ON_THEIR_SIDE";
    case NUM_PAWN_TARGETS_IN_CENTER: return "NUM_PAWN_TARGETS_IN_CENTER";
    case NUM_KNIGHT_TARGETS_IN_CENTER: return "NUM_KNIGHT_TARGETS_IN_CENTER";
    case NUM_BISHOP_TARGETS_IN_CENTER: return "NUM_BISHOP_TARGETS_IN_CENTER";
    case NUM_ROOK_TARGETS_IN_CENTER: return "NUM_ROOK_TARGETS_IN_CENTER";
    case NUM_QUEEN_TARGETS_IN_CENTER: return "NUM_QUEEN_TARGETS_IN_CENTER";
    case NUM_PAWNS_4th_RANK: return "NUM_PAWNS_4th_RANK";
    case NUM_PAWNS_5th_RANK: return "NUM_PAWNS_5th_RANK";
    case NUM_PAWNS_6th_RANK: return "NUM_PAWNS_6th_RANK";
    case NUM_KNIGHTS_ON_EDGE: return "NUM_KNIGHTS_ON_EDGE";
    case NUM_SCARY_BISHOPS: return "NUM_SCARY_BISHOPS";
    case NUM_SCARY_ROOKS: return "NUM_SCARY_ROOKS";
    case ROOKS_ON_OPEN_FILE: return "ROOKS_ON_OPEN_FILE";
    case ROOKS_ON_SEMI_OPEN_FILE: return "ROOKS_ON_SEMI_OPEN_FILE";
    case CONNECTED_ROOKS: return "CONNECTED_ROOKS";
    case PINNED_MINORS: return "PINNED_MINORS";
    case BACK_RANK_MATE_THREAT: return "BACK_RANK_MATE_THREAT";
    case KING_TROPISM: return "KING_TROPISM";
    case OPPOSITION: return "OPPOSITION";
    case ONLY_HAVE_PAWNS: return "ONLY_HAVE_PAWNS";
    case LONELY_KING_DIST_TO_EDGE: return "LONELY_KING_DIST_TO_EDGE";
    case LONELY_KING_DIST_TO_CORNER: return "LONELY_KING_DIST_TO_CORNER";
    case LONELY_KING_DIST_TO_KING: return "LONELY_KING_DIST_TO_KING";
    case EF_COUNT: return "EF_COUNT";
    default: return "UNKNOWN";
  }
}

static const Bitboard kWhiteRanks[8] = {
  kRanks[RANK_1],
  kRanks[RANK_2],
  kRanks[RANK_3],
  kRanks[RANK_4],
  kRanks[RANK_5],
  kRanks[RANK_6],
  kRanks[RANK_7],
  kRanks[RANK_8],
};

static const Bitboard kBlackRanks[8] = {
  kRanks[RANK_8],
  kRanks[RANK_7],
  kRanks[RANK_6],
  kRanks[RANK_5],
  kRanks[RANK_4],
  kRanks[RANK_3],
  kRanks[RANK_2],
  kRanks[RANK_1],
};

const int kMaxEarliness = 18;

template<Color US>
void pos2features(const Position& pos, const Threats& threats, int8_t *out) {
  static constexpr Color THEM = opposite_color<US>();
  static constexpr Direction FORWARD = US == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
  static constexpr Direction FORWARDx2 = US == Color::WHITE ? Direction::NORTHx2 : Direction::SOUTHx2;
  static constexpr Direction BACKWARD = US == Color::WHITE ? Direction::SOUTH : Direction::NORTH;
  static constexpr Direction BACKWARDx2 = US == Color::WHITE ? Direction::SOUTHx2 : Direction::NORTHx2;
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
  const Bitboard ourPieces = ourMinors | ourHeavies;
  const Bitboard theirPieces = theirMinors | theirHeavies;

  const Bitboard ourTargets = US == Color::WHITE ? threats.whiteTargets : threats.blackTargets;
  const Bitboard theirTargets = US == Color::WHITE ? threats.blackTargets : threats.whiteTargets;

  const auto *ourRanks = US == Color::WHITE ? kWhiteRanks : kBlackRanks;
  const auto *theirRanks = US == Color::WHITE ? kBlackRanks : kWhiteRanks;
  const PawnAnalysis<US> pawnAnalysis(pos);

  out[EF::EARLINESS] = 0;
  out[EF::EARLINESS] += std::popcount(ourMinors | ourRooks) + std::popcount(theirMinors | theirRooks);
  out[EF::EARLINESS] += (std::popcount(ourQueens) + std::popcount(theirQueens)) * 3;

  out[EF::PAWNS] = std::popcount(ourPawns) - std::popcount(theirPawns);
  out[EF::KNIGHTS] = std::popcount(ourKnights) - std::popcount(theirKnights);
  out[EF::BISHOPS] = std::popcount(ourBishops) - std::popcount(theirBishops);
  out[EF::ROOKS] = std::popcount(ourRooks) - std::popcount(theirRooks);
  out[EF::QUEENS] = std::popcount(ourQueens) - std::popcount(theirQueens);

  // Only becomes non-zero after ~half of material is traded off. Useful
  // for lategame-only features (e.g. king tropism).
  const int lateness = kMaxEarliness - out[EF::EARLINESS];
  const int veryLateness = std::max(0, lateness - kMaxEarliness / 2);

  out[EF::KING_ON_BACK_RANK] = std::popcount(ourRanks[0] & ourKings) - std::popcount(theirRanks[0] & theirKings);
  if (US == Color::WHITE) {
    out[EF::KING_ACTIVE] = (ourKingSq / 8 < 5) - (theirKingSq / 8 > 2);
  } else {
    out[EF::KING_ACTIVE] = (ourKingSq / 8 > 2) - (theirKingSq / 8 < 5);
  }
  out[EF::KING_ACTIVE] *= veryLateness;

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

  const Bitboard theirSide = ourRanks[4] | ourRanks[5] | ourRanks[6] | ourRanks[7];
  const Bitboard ourSide = ~theirSide;

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

  out[EF::NUM_PAWNS_4th_RANK] = std::popcount(ourPawns & ourRanks[3]) - std::popcount(theirPawns & theirRanks[3]);
  out[EF::NUM_PAWNS_5th_RANK] = std::popcount(ourPawns & ourRanks[4]) - std::popcount(theirPawns & theirRanks[4]);
  out[EF::NUM_PAWNS_6th_RANK] = std::popcount(ourPawns & ourRanks[5]) - std::popcount(theirPawns & theirRanks[5]);
  out[EF::NUM_KNIGHTS_ON_EDGE] = std::popcount(ourKnights & kOuterRing) - std::popcount(theirKnights & kOuterRing);

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

  {
    // An inexact estimate of back-rank threats.
    // Our king on back rank && can only legally move on the back rank && their rook can come to the back rank.
    const Bitboard ourRookTargets = US == Color::WHITE ? threats.whiteRookTargets : threats.blackRookTargets;
    const Bitboard theirRookTargets = US == Color::BLACK ? threats.whiteRookTargets : threats.blackRookTargets;
    const Bitboard ourKingEscapes = compute_king_targets<US>(pos, ourKingSq) & ~(threats.badFor<coloredPiece<US>(Piece::KING)>() | ourPawns);
    const Bitboard theirKingEscapes = compute_king_targets<THEM>(pos, theirKingSq) & ~(threats.badFor<coloredPiece<THEM>(Piece::KING)>() | theirPawns);
    const bool backRankMateThreatAgainstUs = (ourKings & ourRanks[0]) && ((ourKingEscapes & ourRanks[0]) == ourKingEscapes) && (theirRookTargets & ourRanks[0]);
    const bool backRankMateThreatAgainstThem = (theirKings & theirRanks[0]) && ((theirKingEscapes & theirRanks[0]) == theirKingEscapes) && (ourRookTargets & theirRanks[0]);
    out[EF::BACK_RANK_MATE_THREAT] = backRankMateThreatAgainstUs - backRankMateThreatAgainstThem;
  }

  {
    // https://www.chessprogramming.org/King_Pawn_Tropism
    // Penalize king for being far away from pawns. Positive score is *good* for the mover.
    // Note: passed pawns are implicility prioritized, since we consider distance to the passed
    //       pawn, and the two squares ahead of it (so they're weighted 3x).
    const Bitboard passedPawns = pawnAnalysis.ourPassedPawns | pawnAnalysis.theirPassedPawns;
    const Bitboard otherPawns = (ourPawns | theirPawns) & ~passedPawns;
    const Bitboard aheadOfPassedPawns = passedPawns | shift<FORWARD>(pawnAnalysis.ourPassedPawns) | shift<BACKWARD>(pawnAnalysis.theirPassedPawns)
    | shift<FORWARDx2>(pawnAnalysis.ourPassedPawns) | shift<BACKWARDx2>(pawnAnalysis.theirPassedPawns);
    int tropism = 0;
    for (int i = 0; i < 15; ++i) {
      tropism += std::popcount(aheadOfPassedPawns & kManhattanDist[i][ourKingSq]);
      tropism += std::popcount(otherPawns & kManhattanDist[i][ourKingSq]);

      tropism -= std::popcount(kManhattanDist[i][theirKingSq] & aheadOfPassedPawns);
      tropism -= std::popcount(kManhattanDist[i][theirKingSq] & otherPawns);
    }
    tropism *= veryLateness;
    out[EF::KING_TROPISM] = std::min(std::max(-127, tropism), 127);
  }

  const bool weOnlyHavePawns = (ourPieces == ourKings);
  const bool theyOnlyHavePawns = (theirPieces == theirKings);

  // A lot of our "mop up" features rely on "ONLY_HAVE_PAWNS" so it's good to learn
  // a bias term so our mop up weights are meaningful.
  out[EF::ONLY_HAVE_PAWNS] = weOnlyHavePawns - theyOnlyHavePawns;

  {
    // Note: it's our turn, so if anyone has the opposition, it is our opponent.
    int dx = std::abs(ourKingSq % 8 - theirKingSq % 8);
    int dy = std::abs(ourKingSq / 8 - theirKingSq / 8);
    out[EF::OPPOSITION] = weOnlyHavePawns && ((dx == 0 && dy == 2) || (dx == 2 && dy == 0));
  }

  // Stay away from edge if you only have a king. This (e.g.) helps us checkmate with 2 bishops.
  out[EF::LONELY_KING_DIST_TO_EDGE] = kDistToEdge[ourKingSq] * weOnlyHavePawns * !theyOnlyHavePawns - kDistToEdge[theirKingSq] * theyOnlyHavePawns * !weOnlyHavePawns;
  out[EF::LONELY_KING_DIST_TO_CORNER] = kDistToCorner[ourKingSq] * weOnlyHavePawns * !theyOnlyHavePawns - kDistToCorner[theirKingSq] * theyOnlyHavePawns * !weOnlyHavePawns;

  // Stay away from enemy king if you only have a king. This (e.g.) helps us checkmate with 2 bishops.
  const int kingDistance = king_dist(ourKingSq, theirKingSq);
  out[EF::LONELY_KING_DIST_TO_KING] = kingDistance * weOnlyHavePawns * !theyOnlyHavePawns - kingDistance * theyOnlyHavePawns * !weOnlyHavePawns;
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
    int32_t late = bias[0];
    int32_t early = bias[1];
    for (size_t i = 0; i < EF::EF_COUNT; ++i) {
      late += x[i] * weights(0, i);
      early += x[i] * weights(1, i);
    }
    int32_t earliness = x[EF::EARLINESS];
    int32_t r = (early * earliness + late * (kMaxEarliness - earliness)) / kMaxEarliness;
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
