
#ifndef SRC_EVAL_BYHAND_BYHAND_H
#define SRC_EVAL_BYHAND_BYHAND_H

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include "../../game/Position.h"
#include "../../game/Utils.h"
#include "../../game/BoardListener.h"
#include "../../game/Geometry.h"
#include "../../game/movegen/bishops.h"
#include "../Evaluator.h"
#include "../pst/PieceSquareEvaluator.h"
#include "../ColoredEvaluation.h"
#include "../pst/PieceSquareEvaluator.h"
#include "../nnue/Nnue.h"
#include "../PawnAnalysis.h"

namespace ChessEngine {

namespace ByHand {

/* TODO
  - Rook behind passed pawn
  - isolated and/or doubled and/or backward pawns *on semiopen files* are worse
  - double pawns get worse as rooks/queens come off the board
  - if/when we add pawn majority/minority logic, double pawns in the majority side are worse than normal doubled pawns
  - distinguish between doubled pawns in a defensive triangle and doubled pawns adjacent to each other?
  - distinguish between pawn types on different files (e.g. doubled pawn on the edge vs in the center)
  - num pawns that can safely advance? on each row?
*/
enum EF {
  EARLINESS,

  PAWNS,  // 1
  KNIGHTS,
  BISHOPS,
  ROOKS,
  QUEENS,

  KING_ON_BACK_RANK,  // 6
  KING_ACTIVE,  // 7
  THREATS_NEAR_KING_2,  // 8
  THREATS_NEAR_KING_3,  // 9
  PASSED_PAWNS,  // 10
  PASSED_PAWNS_7TH_RANK,  // 11
  PASSED_PAWNS_6TH_RANK,  // 12

  ISOLATED_PAWNS,  // 13
  DOUBLED_PAWNS,  // 14
  DOUBLE_ISOLATED_PAWNS,  // 15

  HANGING_PAWN,  // 16
  HANGING_KNIGHT,  // 17
  HANGING_BISHOP,  // 18
  HANGING_ROOK,  // 19
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

  // Pawn that cannot be defended by another pawn, and
  // whose advancement is prevented by an enemy pawn.
  BACKWARD_PAWN,

  // Backward pawn on a file where the opponent has no pawn.
  STRAGGLER_PAWN,

  // A pawn that can become a passed pawn with the help
  // of its neighbors.
  CANDIDATE_PASSED_PAWN,

  // 8/5pk1/7p/2q5/5p2/5Q2/5KP1/8 w - - 16 89
  // White's promotion square is being fought over, but
  // black's is dominated.

  NvN,
  BvN,
  RvN,
  BvB_opposite,
  BvB_same,
  BvR,
  RvR,

  PAWN_FORKS_Q,
  PAWN_FORKS_R,
  PAWN_FORKS,
  PAWN_FORK_THREATS,
  KNIGHT_FORKS_Q,
  KNIGHT_FORKS_R,
  KNIGHT_FORKS,
  KNIGHT_FORK_THREATS,

  TRAPPED_KNIGHT,
  TRAPPED_BISHOP,
  TRAPPED_ROOK,
  TRAPPED_QUEEN,

  CURRENT_KING_HOME_QUALITY,
  KINGSIDE_HOME_QUALITY,
  QUEENSIDE_HOME_QUALITY,
  POTENTIAL_HOME_QUALITY,
  PAWN_WEST_OF_KING,
  PAWN_AHEAD_OF_KING,
  PAWN_EAST_OF_KING,
  PAWN_WEST_OF_KING_2,
  PAWN_AHEAD_OF_KING_2,
  PAWN_EAST_OF_KING_2,

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
    case BACKWARD_PAWN: return "BACKWARD_PAWN";
    case STRAGGLER_PAWN: return "STRAGGLER_PAWN";
    case CANDIDATE_PASSED_PAWN: return "CANDIDATE_PASSED_PAWN";
    case EF_COUNT: return "EF_COUNT";
    case NvN: return "NvN";
    case BvN: return "BvN";
    case RvN: return "RvN";
    case BvB_opposite: return "BvB_opposite";
    case BvB_same: return "BvB_same";
    case BvR: return "BvR";
    case RvR: return "RvR";
    case PAWN_FORKS_Q: return "PAWN_FORKS_Q";
    case PAWN_FORKS_R: return "PAWN_FORKS_R";
    case PAWN_FORKS: return "PAWN_FORKS";
    case PAWN_FORK_THREATS: return "PAWN_FORK_THREATS";
    case KNIGHT_FORKS: return "KNIGHT_FORKS";
    case KNIGHT_FORKS_Q: return "KNIGHT_FORKS_Q";
    case KNIGHT_FORKS_R: return "KNIGHT_FORKS_R";
    case KNIGHT_FORK_THREATS: return "KNIGHT_FORK_THREATS";
    case TRAPPED_KNIGHT: return "TRAPPED_KNIGHT";
    case TRAPPED_BISHOP: return "TRAPPED_BISHOP";
    case TRAPPED_ROOK: return "TRAPPED_ROOK";
    case TRAPPED_QUEEN: return "TRAPPED_QUEEN";
    case CURRENT_KING_HOME_QUALITY: return "CURRENT_KING_HOME_QUALITY";
    case KINGSIDE_HOME_QUALITY: return "KINGSIDE_HOME_QUALITY";
    case QUEENSIDE_HOME_QUALITY: return "QUEENSIDE_HOME_QUALITY";
    case POTENTIAL_HOME_QUALITY: return "POTENTIAL_HOME_QUALITY";
    case PAWN_WEST_OF_KING: return "PAWN_WEST_OF_KING";
    case PAWN_AHEAD_OF_KING: return "PAWN_AHEAD_OF_KING";
    case PAWN_EAST_OF_KING: return "PAWN_EAST_OF_KING";
    case PAWN_WEST_OF_KING_2: return "PAWN_WEST_OF_KING_2";
    case PAWN_AHEAD_OF_KING_2: return "PAWN_AHEAD_OF_KING_2";
    case PAWN_EAST_OF_KING_2: return "PAWN_EAST_OF_KING_2";
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

template<Color KING_COLOR>
inline int kingHomeQuality(const Position& pos, const SafeSquare kingSq) {
  static constexpr Direction FORWARD = KING_COLOR == Color::WHITE ? Direction::NORTH : Direction::SOUTH;
  static constexpr Direction BACKWARD = KING_COLOR == Color::WHITE ? Direction::SOUTH : Direction::NORTH;
  static constexpr Direction FORWARD_EAST = KING_COLOR == Color::WHITE ? Direction::NORTH_EAST : Direction::SOUTH_EAST;
  static constexpr Direction FORWARD_WEST = KING_COLOR == Color::WHITE ? Direction::NORTH_WEST : Direction::SOUTH_WEST;

  const Bitboard ourKings = bb(kingSq);
  const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<KING_COLOR, Piece::PAWN>()];
  const Bitboard ourPawnsShiftedDown = shift<BACKWARD>(ourPawns);
  
  const int distFromBackRank = KING_COLOR == Color::WHITE ? 7 - kingSq / 8 : kingSq / 8;

  // +2 for pawns immediately in front of the king, +1 for pawns one step further removed.
  int pawnQuality = 0;
  pawnQuality += (shift<FORWARD>(ourKings) & ourPawns) ? 2 : 0;
  pawnQuality += (shift<FORWARD>(ourKings) & ourPawnsShiftedDown) ? 1 : 0;
  pawnQuality += (shift<FORWARD_EAST>(ourKings) & ourPawns) ? 2 : 0;
  pawnQuality += (shift<FORWARD_EAST>(ourKings) & ourPawnsShiftedDown) ? 1 : 0;
  pawnQuality += (shift<FORWARD_WEST>(ourKings) & ourPawns) ? 2 : 0;
  pawnQuality += (shift<FORWARD_WEST>(ourKings) & ourPawnsShiftedDown) ? 1 : 0;

  int homeQuality = (pawnQuality * 4) / (distFromBackRank + 1);
  return homeQuality;
}

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
  const Bitboard ourPieces = ourMinors | ourHeavies;  // Exclude pawns.
  const Bitboard theirPieces = theirMinors | theirHeavies;
  const Bitboard ourMen = pos.colorBitboards_[US];
  const Bitboard theirMen = pos.colorBitboards_[THEM];
  const Bitboard everyone = ourMen | theirMen;

  const Bitboard ourTargets = US == Color::WHITE ? threats.whiteTargets : threats.blackTargets;
  const Bitboard theirTargets = US == Color::WHITE ? threats.blackTargets : threats.whiteTargets;

  const auto *ourRanks = US == Color::WHITE ? kWhiteRanks : kBlackRanks;
  const auto *theirRanks = US == Color::WHITE ? kBlackRanks : kWhiteRanks;
  const PawnAnalysis<US> pawnAnalysis(pos);

  out[EF::EARLINESS] = 0;
  out[EF::EARLINESS] += std::popcount(ourMinors | ourRooks) + std::popcount(theirMinors | theirRooks);
  out[EF::EARLINESS] += (std::popcount(ourQueens) + std::popcount(theirQueens)) * 3;
  out[EF::EARLINESS] = std::min<int>(out[EF::EARLINESS], kMaxEarliness);

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

  Bitboard ourPawnTargets = US == Color::WHITE ? threats.whitePawnTargets : threats.blackPawnTargets;
  ourPawnTargets &= ~threats.badForOur<US>(Piece::PAWN);
  Bitboard theirPawnTargets = US == Color::WHITE ? threats.blackPawnTargets : threats.whitePawnTargets;
  theirPawnTargets &= ~threats.badForOur<THEM>(Piece::PAWN);
  Bitboard ourKnightTargets = US == Color::WHITE ? threats.whiteKnightTargets : threats.blackKnightTargets;
  ourKnightTargets &= ~threats.badForOur<US>(Piece::KNIGHT);
  Bitboard theirKnightTargets = US == Color::WHITE ? threats.blackKnightTargets : threats.whiteKnightTargets;
  theirKnightTargets &= ~threats.badForOur<THEM>(Piece::KNIGHT);
  Bitboard ourBishopTargets = US == Color::WHITE ? threats.whiteBishopTargets : threats.blackBishopTargets;
  ourBishopTargets &= ~threats.badForOur<US>(Piece::BISHOP);
  Bitboard theirBishopTargets = US == Color::WHITE ? threats.blackBishopTargets : threats.whiteBishopTargets;
  theirBishopTargets &= ~threats.badForOur<THEM>(Piece::BISHOP);
  Bitboard ourRookTargets = US == Color::WHITE ? threats.whiteRookTargets : threats.blackRookTargets;
  ourRookTargets &= ~threats.badForOur<US>(Piece::ROOK);
  Bitboard theirRookTargets = US == Color::WHITE ? threats.blackRookTargets : threats.whiteRookTargets;
  theirRookTargets &= ~threats.badForOur<THEM>(Piece::ROOK);
  Bitboard ourQueenTargets = US == Color::WHITE ? threats.whiteQueenTargets : threats.blackQueenTargets;
  ourQueenTargets &= ~threats.badForOur<US>(Piece::QUEEN);
  Bitboard theirQueenTargets = US == Color::WHITE ? threats.blackQueenTargets : threats.whiteQueenTargets;
  theirQueenTargets &= ~threats.badForOur<THEM>(Piece::QUEEN);

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
  const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(ourBishops, ourBlockadedPawns) & ~threats.badForOur<US>(Piece::BISHOP);
  const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(theirBishops, theirBlockadedPawns) & ~threats.badForOur<THEM>(Piece::BISHOP);

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

  const Bitboard ourPawnTargetsFixed = US == Color::WHITE ? threats.whitePawnTargets : threats.blackPawnTargets;
  const Bitboard theirPawnTargetsFixed = US == Color::WHITE ? threats.blackPawnTargets : threats.whitePawnTargets;
  const Bitboard aheadOfOurPawns = US == Color::WHITE ? northFill(ourPawns) : southFill(ourPawns);
  const Bitboard aheadOfTheirPawns = US == Color::WHITE ? southFill(theirPawns) : northFill(theirPawns);

  const Bitboard eventuallyAttackedByOurPawns1 = shift<EAST>(aheadOfOurPawns);
  const Bitboard eventuallyAttackedByOurPawns2 = shift<WEST>(aheadOfOurPawns);
  const Bitboard eventuallyAttackedByOurPawns = eventuallyAttackedByOurPawns1 | eventuallyAttackedByOurPawns2;

  const Bitboard eventuallyAttackedByTheirPawns1 = shift<EAST>(aheadOfTheirPawns);
  const Bitboard eventuallyAttackedByTheirPawns2 = shift<WEST>(aheadOfTheirPawns);
  const Bitboard eventuallyAttackedByTheirPawns = eventuallyAttackedByTheirPawns1 | eventuallyAttackedByTheirPawns2;

  const Bitboard filesWithOurPawns = US == Color::WHITE ? southFill(aheadOfOurPawns) : northFill(aheadOfOurPawns);
  const Bitboard filesWithTheirPawns = US == Color::WHITE ? northFill(aheadOfTheirPawns) : southFill(aheadOfTheirPawns);

  // Backward pawn: a pawn that cannot be defended by another pawn, and whose advancement
  // is prevented by an enemy pawn.
  const Bitboard ourBackwardPawns = ourPawns & shift<BACKWARD>(theirPawnTargetsFixed) & ~eventuallyAttackedByOurPawns;
  const Bitboard theirBackwardPawns = theirPawns & shift<FORWARD>(ourPawnTargetsFixed) & ~eventuallyAttackedByTheirPawns;
  out[EF::BACKWARD_PAWN] = std::popcount(ourBackwardPawns) - std::popcount(theirBackwardPawns);

  // Straggler pawn: a backward pawn on a file where the opponent has no pawn.
  const Bitboard ourStragglers = ourBackwardPawns & ~(filesWithTheirPawns);
  const Bitboard theirStragglers = theirBackwardPawns & ~(filesWithOurPawns);
  out[EF::STRAGGLER_PAWN] = std::popcount(ourStragglers) - std::popcount(theirStragglers);

  // A pawn that can become a passed pawn with the help
  // of its neighbors.
  // First we determine which files we have with at least as many neighboring pawns as the opponent.
  const Bitboard ourCandidateFiles = (eventuallyAttackedByOurPawns & ~eventuallyAttackedByTheirPawns) | (eventuallyAttackedByOurPawns1 & eventuallyAttackedByOurPawns2) | (~fatten(aheadOfTheirPawns));
  const Bitboard theirCandidateFiles = (eventuallyAttackedByTheirPawns & ~eventuallyAttackedByOurPawns) | (eventuallyAttackedByTheirPawns1 & eventuallyAttackedByTheirPawns2) | (~fatten(aheadOfOurPawns));
  // Then we check if we have any pawns on those files that can advance (i.e. are not blocked by an enemy pawn).
  // We mask everything to the last rank since "ourCandidateFiles" is guaranteed to extend to there, but
  // not any other rank.
  const Bitboard ourCandidatePawnFiles = (filesWithOurPawns & ~filesWithTheirPawns & ourCandidateFiles) & ourRanks[7];
  const Bitboard theirCandidatePawnFiles = (filesWithTheirPawns & ~filesWithOurPawns & theirCandidateFiles) & theirRanks[7];
  out[EF::CANDIDATE_PASSED_PAWN] = std::popcount(ourCandidatePawnFiles) - std::popcount(theirCandidatePawnFiles);

  const bool weHaveOnePiece = std::popcount(ourPieces & ~ourKings) == 1;
  const bool theyHaveOnePiece = std::popcount(theirPieces & ~theirKings) == 1;
  const bool weOnlyHaveOneKnight = weHaveOnePiece && (ourKnights | ourKings) == ourPieces;
  const bool theyOnlyHaveOneKnight = theyHaveOnePiece && (theirKnights | theirKings) == theirPieces;
  const bool weOnlyHaveOneBishop = weHaveOnePiece && (ourBishops | ourKings) == ourPieces;
  const bool theyOnlyHaveOneBishop = theyHaveOnePiece && (theirBishops | theirKings) == theirPieces;
  const bool weOnlyHaveOneRook = weHaveOnePiece && (ourRooks | ourKings) == ourPieces;
  const bool theyOnlyHaveOneRook = theyHaveOnePiece && (theirRooks | theirKings) == theirPieces;
  out[EF::NvN] = weOnlyHaveOneKnight && theyOnlyHaveOneKnight;
  out[EF::BvN] = (weOnlyHaveOneBishop && theyOnlyHaveOneKnight) - (weOnlyHaveOneKnight && theyOnlyHaveOneBishop);
  out[EF::RvN] = (weOnlyHaveOneRook && theyOnlyHaveOneKnight) - (weOnlyHaveOneKnight && theyOnlyHaveOneRook);
  out[EF::BvB_opposite] = (weOnlyHaveOneBishop && theyOnlyHaveOneBishop) && ((ourBishops & kWhiteSquares) != (theirBishops & kWhiteSquares));
  out[EF::BvB_same] = (weOnlyHaveOneBishop && theyOnlyHaveOneBishop) && ((ourBishops & kWhiteSquares) == (theirBishops & kWhiteSquares));
  out[EF::BvR] = (weOnlyHaveOneBishop && theyOnlyHaveOneRook) - (weOnlyHaveOneRook && theyOnlyHaveOneBishop);
  out[EF::RvR] = (weOnlyHaveOneRook && theyOnlyHaveOneRook);

  // Note: this "overcounts" true forks, since two different pawns can attack two different pieces, but this is probably a good thing.
  out[EF::PAWN_FORKS_Q] = (std::popcount(ourPawnTargets & theirRoyalty) >= 2) - (std::popcount(theirPawnTargets & ourRoyalty) >= 2);
  out[EF::PAWN_FORKS_R] = (std::popcount(ourPawnTargets & theirHeavies) >= 2) - (std::popcount(theirPawnTargets & ourHeavies) >= 2);
  out[EF::PAWN_FORKS] = (std::popcount(ourPawnTargets & theirPieces) >= 2) - (std::popcount(theirPawnTargets & ourPieces) >= 2);

  // Squares our pawns can safely move to, that attack 2+ enemy pieces.
  const Bitboard ourPawnForkThreats = shift<BACKWARD_EAST>(theirPieces) & shift<BACKWARD_WEST>(theirPieces) & ~threats.badFor<coloredPiece<US>(Piece::PAWN)>() & shift<FORWARD>(ourPawns) & ~everyone;
  const Bitboard theirPawnForkThreats = shift<FORWARD_EAST>(ourPieces) & shift<FORWARD_WEST>(ourPieces) & ~threats.badFor<coloredPiece<THEM>(Piece::PAWN)>() & shift<BACKWARD>(theirPawns) & ~everyone;
  out[EF::PAWN_FORK_THREATS] = std::popcount(ourPawnForkThreats) - std::popcount(theirPawnForkThreats);

  // Overcounts true knight forks, similar to EF::PAWN_FORKS.
  out[EF::KNIGHT_FORKS_Q] = (std::popcount(ourKnightTargets & theirRoyalty) >= 2) - (std::popcount(theirKnightTargets & ourRoyalty) >= 2);
  out[EF::KNIGHT_FORKS_R] = (std::popcount(ourKnightTargets & theirHeavies) >= 2) - (std::popcount(theirKnightTargets & ourHeavies) >= 2);
  out[EF::KNIGHT_FORKS] = (std::popcount(ourKnightTargets & theirPieces) >= 2) - (std::popcount(theirKnightTargets & ourPieces) >= 2);

  // Squares our knights can move to, that attack rooks or royalty.
  const Bitboard ourForkTargets = at_least_two(
    kKnightMoves[to_unsafe_square(theirKingSq)],
    kKnightMoves[lsb_or_none(theirQueens)],
    kKnightMoves[lsb_or_none(theirRooks)],
    kKnightMoves[lsb_or_none(theirRooks & (theirRooks - 1))]
  ) & ~threats.badFor<coloredPiece<US, Piece::KNIGHT>()>() & ~ourMen;
  const Bitboard theirForkTargets = at_least_two(
    kKnightMoves[to_unsafe_square(ourKingSq)],
    kKnightMoves[lsb_or_none(ourQueens)],
    kKnightMoves[lsb_or_none(ourRooks)],
    kKnightMoves[lsb_or_none(ourRooks & (ourRooks - 1))]
  ) & ~threats.badFor<coloredPiece<US, Piece::KNIGHT>()>() & ~ourMen;
  out[EF::KNIGHT_FORK_THREATS] = std::popcount(ourKnightTargets & ourForkTargets) - std::popcount(theirKnightTargets & theirForkTargets);

  // This implementation is a little hacky:
  // 1) It doesn't work if you have two bishops and one of them is trapped.
  // 2) It counts a bishops on the back rank with no safe moves as "trapped",
  //    since these pices are not typically in danger of being captured.
  out[EF::TRAPPED_KNIGHT] = (std::popcount(ourKnightTargets) == 0 && ((ourKnights & ourRanks[0]) != 0)) - (std::popcount(theirKnightTargets) == 0 && ((theirKnights & theirRanks[0]) != 0));
  out[EF::TRAPPED_BISHOP] = (std::popcount(ourBishopTargets) == 0 && ((ourBishops & ourRanks[0]) != 0)) - (std::popcount(theirBishopTargets) == 0 && ((theirBishops & theirRanks[0]) != 0));
  out[EF::TRAPPED_ROOK] = (std::popcount(ourRookTargets) == 0 && ((ourRooks & ourRanks[0]) != 0)) - (std::popcount(theirRookTargets) == 0 && ((theirRooks & theirRanks[0]) != 0));
  out[EF::TRAPPED_QUEEN] = (std::popcount(ourQueenTargets) == 0 && ((ourQueens & ourRanks[0]) != 0)) - (std::popcount(theirQueenTargets) == 0 && ((theirQueens & theirRanks[0]) != 0));

  const bool canWeCastleKingside = pos.currentState_.castlingRights & (
    US == Color::WHITE ? ChessEngine::kCastlingRights_WhiteKing : ChessEngine::kCastlingRights_BlackKing
  );
  const bool canWeCastleQueenside = pos.currentState_.castlingRights & (
    US == Color::WHITE ? ChessEngine::kCastlingRights_WhiteQueen : ChessEngine::kCastlingRights_BlackQueen
  );
  const bool canTheyCastleKingside = pos.currentState_.castlingRights & (
    THEM == Color::WHITE ? ChessEngine::kCastlingRights_WhiteKing : ChessEngine::kCastlingRights_BlackKing
  );
  const bool canTheyCastleQueenside = pos.currentState_.castlingRights & (
    THEM == Color::WHITE ? ChessEngine::kCastlingRights_WhiteQueen : ChessEngine::kCastlingRights_BlackQueen
  );
  const bool canWeCastle = canWeCastleKingside || canWeCastleQueenside;
  const bool canTheyCastle = canTheyCastleKingside || canTheyCastleQueenside;

  // G or H file, or can castle kingside.
  const bool areWeOnKingside = (ourKingSq % 8) > 5 || canWeCastleKingside;
  const bool areTheyOnKingside = (theirKingSq % 8) > 5 || canTheyCastleKingside;
  // A, B, or C file, or can castle queenside.
  const bool areWeOnQueenside = (ourKingSq % 8) <= 2 || canWeCastleQueenside;
  const bool areTheyOnQueenside = (theirKingSq % 8) <= 2 || canTheyCastleQueenside;

  const int ourCurrentHomeQuality = kingHomeQuality<US>(pos, ourKingSq);
  const int ourKingsideHomeQuality = (
    kingHomeQuality<US>(pos, US == Color::WHITE ? SafeSquare::SG1 : SafeSquare::SG8)
  ) * areWeOnKingside;
  const int ourQueensideHomeQuality = (
    kingHomeQuality<US>(pos, US == Color::WHITE ? SafeSquare::SB1 : SafeSquare::SB8)
  ) * areWeOnQueenside;
  const int theirCurrentHomeQuality = kingHomeQuality<THEM>(pos, theirKingSq);
  const int theirKingsideHomeQuality = (
    kingHomeQuality<THEM>(pos, THEM == Color::WHITE ? SafeSquare::SG1 : SafeSquare::SG8)
  ) * areTheyOnKingside;
  const int theirQueensideHomeQuality = (
    kingHomeQuality<THEM>(pos, THEM == Color::WHITE ? SafeSquare::SB1 : SafeSquare::SB8)
  ) * areTheyOnQueenside;

  const int ourPotentialHomeQuality = std::max(ourCurrentHomeQuality, std::max(ourKingsideHomeQuality, ourQueensideHomeQuality));
  const int theirPotentialHomeQuality = std::max(theirCurrentHomeQuality, std::max(theirKingsideHomeQuality, theirQueensideHomeQuality));

  out[EF::CURRENT_KING_HOME_QUALITY] = ourCurrentHomeQuality - theirCurrentHomeQuality;
  out[EF::KINGSIDE_HOME_QUALITY] = ourKingsideHomeQuality - theirKingsideHomeQuality;
  out[EF::QUEENSIDE_HOME_QUALITY] = ourQueensideHomeQuality - theirQueensideHomeQuality;
  out[EF::POTENTIAL_HOME_QUALITY] = ourPotentialHomeQuality - theirPotentialHomeQuality;

  out[EF::PAWN_WEST_OF_KING] = std::popcount(ourPawns & shift<FORWARD_WEST>(ourKings)) * (!canWeCastle) - std::popcount(theirPawns & shift<BACKWARD_WEST>(theirKings)) * (!canTheyCastle);
  out[EF::PAWN_AHEAD_OF_KING] = std::popcount(ourPawns & shift<FORWARD>(ourKings)) * (!canWeCastle) - std::popcount(theirPawns & shift<BACKWARD>(theirKings)) * (!canTheyCastle);
  out[EF::PAWN_EAST_OF_KING] = std::popcount(ourPawns & shift<FORWARD_EAST>(ourKings)) * (!canWeCastle) - std::popcount(theirPawns & shift<BACKWARD_EAST>(theirKings)) * (!canTheyCastle);
  out[EF::PAWN_WEST_OF_KING_2] = std::popcount(
    ourPawns & shift<FORWARD>(shift<FORWARD_WEST>(ourKings))) * (!canWeCastle) - std::popcount(
      theirPawns & shift<BACKWARD>(shift<BACKWARD_WEST>(theirKings))) * (!canTheyCastle);
  out[EF::PAWN_AHEAD_OF_KING_2] = std::popcount(
    ourPawns & shift<FORWARD>(shift<FORWARD>(ourKings))) * (!canWeCastle) - std::popcount(
      theirPawns & shift<BACKWARD>(shift<BACKWARD>(theirKings))) * (!canTheyCastle);
  out[EF::PAWN_EAST_OF_KING_2] = std::popcount(
    ourPawns & shift<FORWARD>(shift<FORWARD_EAST>(ourKings))) * (!canWeCastle) - std::popcount(
      theirPawns & shift<BACKWARD>(shift<BACKWARD_EAST>(theirKings))) * (!canTheyCastle);
}

struct ByHandEvaluator : public PieceSquareEvaluator {
  NNUE::Matrix<2, EF::EF_COUNT, int16_t> weights;
  NNUE::Vector<2, int16_t> bias;
  NNUE::Vector<EF::EF_COUNT, int8_t> x;
  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::WHITE> alpha, ColoredEvaluation<Color::WHITE> beta) override {
    return _evaluate<Color::WHITE>(pos, threats, alpha, beta);
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos, const Threats& threats, int plyFromRoot, ColoredEvaluation<Color::BLACK> alpha, ColoredEvaluation<Color::BLACK> beta) override {
    return _evaluate<Color::BLACK>(pos, threats, alpha, beta);
  }

  inline static bool _is_material_draw(const Position& pos) {
    if ((pos.pieceBitboards_[ColoredPiece::WHITE_PAWN] | pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]) == kEmptyBitboard) {
      int whiteMinor = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
      int blackMinor = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT] | pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
      int whiteMajor = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK] | pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
      int blackMajor = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK] | pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);
      return whiteMajor + blackMajor == 0 && whiteMinor <= 1 && blackMinor <= 1;
    }
    return false;
  }

  template<Color US>
  ColoredEvaluation<US> _evaluate(const Position& pos, const Threats& threats, ColoredEvaluation<US> alpha, ColoredEvaluation<US> beta) {
    if (_is_material_draw(pos)) {
      return ColoredEvaluation<US>(kDraw);
    }

    pos2features<US>(pos, threats, x.data_ptr());
    int32_t pst_late = US == Color::WHITE ? this->PieceSquareEvaluator::late : -this->PieceSquareEvaluator::late;
    int32_t pst_early = US == Color::WHITE ? this->PieceSquareEvaluator::early : -this->PieceSquareEvaluator::early;
    int32_t late = bias[0] + pst_late;
    int32_t early = bias[1] + pst_early;
    for (size_t i = 0; i < EF::EF_COUNT; ++i) {
      late += x[i] * weights(0, i);
      early += x[i] * weights(1, i);
    }
    int32_t earliness = x[EF::EARLINESS];
    int32_t r = (early * earliness + late * (kMaxEarliness - earliness)) / kMaxEarliness;

    // This seems to have no effect on playing strength (-0.0008 +/- 0.0021)
    // int32_t halfMoveCounter = pos.currentState_.halfMoveCounter;
    // if (halfMoveCounter >= 20) {
    //   // When halfMoveCounter is 20, multiplies by 1
    //   // When halfMoveCounter is 100, multiplies by 0.5
    //   const int32_t scale = std::clamp(100 - halfMoveCounter, 0, 80);
    //   r = r * (scale + 80) / 160;
    // }
    return ColoredEvaluation<US>(r);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    auto copy = std::make_shared<ByHandEvaluator>();
    copy->weights = this->weights;
    copy->bias = this->bias;
    std::copy(std::begin(this->pstWeights), std::end(this->pstWeights), std::begin(copy->pstWeights));
    return copy;
  }

  void load_from_stream(std::istream& in) {
    weights.load_from_stream(in, "weights");
    bias.load_from_stream(in, "bias");
    NNUE::Matrix<6, 64, int16_t> pstLate;
    pstLate.load_from_stream(in, "pst_late");
    NNUE::Matrix<6, 64, int16_t> pstEarly;
    pstEarly.load_from_stream(in, "pst_early");
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 64; ++j) {
        this->pstWeights[i * 64 + j + 64] = pstEarly(i, j);
        this->pstWeights[i * 64 + j + 8 * 64] = pstLate(i, j);
      }
    }
  }

  std::string to_string() const override {
    return "ByHandEvaluator";
  }

  // void place_piece(ColoredPiece cp, SafeSquare square) override {}
  // void remove_piece(ColoredPiece cp, SafeSquare square) override {}
  // void place_piece(SafeColoredPiece cp, SafeSquare square) override {}
  // void remove_piece(SafeColoredPiece cp, SafeSquare square) override {}
  // void empty() override {}
};

};  // namespace ByHand

}  // namespace ChessEngine

#endif // SRC_EVAL_BYHAND_BYHAND_H
