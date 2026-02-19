#ifndef POSITION_H
#define POSITION_H

#include <cassert>
#include <cstdint>
#include <cstring>

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>

#include "BoardListener.h"
#include "Geometry.h"
#include "Move.h"
#include "Utils.h"
#include "Threats.h"
#include "../StringUtils.h"

namespace ChessEngine {

struct PositionState {
  CastlingRights castlingRights;
  uint8_t halfMoveCounter;
  UnsafeSquare epSquare;
  uint64_t hash;  // Required for 3-move draw.
};

std::ostream& operator<<(std::ostream& stream, const PositionState& state);
struct Position;
std::ostream& operator<<(std::ostream& stream, const Position& pos);

extern uint64_t kZorbristNumbers[kNumColoredPieces][kNumSquares];
extern uint64_t kZorbristCastling[16];
extern uint64_t kZorbristEnpassant[8];
extern uint64_t kZorbristTurn;

void print_zorbrist_debug(uint64_t actual, uint64_t expected);

void initialize_zorbrist();

class Position {
 public:
  Position() : turn_(Color::WHITE) {
    this->turn_ = Color::WHITE;
    this->_empty_();
  };
  Position(const std::string& fen);

  // Copy constructor.
  // We do not copy the board listener; instead we create a dummy listener
  // and set it after copying the position.
  Position(const Position& other)
  : tiles_(other.tiles_),
    pieceBitboards_(other.pieceBitboards_),
    colorBitboards_(other.colorBitboards_),
    states_(other.states_),
    history_(other.history_),
    currentState_(other.currentState_),
    evaluator_(other.evaluator_->clone()),
    wholeMoveCounter_(other.wholeMoveCounter_),
    turn_(other.turn_) {
      this->set_listener(this->evaluator_);
    }

  static Position init();

  std::string fen() const;

  std::string san(Move move) const;

  // ColoredPiece tiles_[kNumSquares];
  TypeSafeArray<ColoredPiece, kNumSquares, SafeSquare> tiles_;
  TypeSafeArray<Bitboard, kNumColoredPieces, ColoredPiece> pieceBitboards_;
  TypeSafeArray<Bitboard, Color::NUM_COLORS, Color> colorBitboards_;

  std::vector<PositionState> states_;
  std::vector<ExtMove> history_;
  PositionState currentState_;
  std::shared_ptr<EvaluatorInterface> evaluator_ = std::make_shared<DummyEvaluator>();

  void set_listener(std::shared_ptr<EvaluatorInterface> listener) {
    evaluator_ = listener;
    evaluator_->empty();
    for (size_t i = 0; i < kNumSquares; ++i) {
        ColoredPiece cp = this->tiles_[SafeSquare(i)];
        if (cp != ColoredPiece::NO_COLORED_PIECE) {
            evaluator_->place_piece(cp, SafeSquare(i));
        }
    }
  }

  // Incremented after a black move.
  uint32_t wholeMoveCounter_;
  Color turn_;

  std::string history_str() const {
    std::string r = "";
    for (const auto& move : history_) {
      r += move.uci() + " ";
    }
    return r;
  }

  void place_piece_(ColoredPiece cp, SafeSquare square);

  void remove_piece_(SafeSquare square);

  bool is_material_draw() const {
    const Bitboard everyone = this->colorBitboards_[Color::WHITE] | this->colorBitboards_[Color::BLACK];
    const Bitboard everyoneButKings = everyone & ~(this->pieceBitboards_[ColoredPiece::WHITE_KING] | this->pieceBitboards_[ColoredPiece::BLACK_KING]);
    const bool isThreeManEndgame = std::popcount(everyone) == 3;
    bool isDraw = false;
    isDraw |= (everyoneButKings == 0);
    isDraw |= (everyoneButKings == (this->pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | this->pieceBitboards_[ColoredPiece::BLACK_KNIGHT])) && isThreeManEndgame;
    isDraw |= (everyoneButKings == (this->pieceBitboards_[ColoredPiece::WHITE_BISHOP] | this->pieceBitboards_[ColoredPiece::BLACK_BISHOP])) && isThreeManEndgame;
    return isDraw;
  }

  inline void increment_piece_map(ColoredPiece cp, SafeSquare sq) {
    this->evaluator_->place_piece(cp, sq);
  }
  inline void decrement_piece_map(ColoredPiece cp, SafeSquare sq) {
    this->evaluator_->remove_piece(cp, sq);
  }

  inline void increment_piece_map(SafeColoredPiece cp, SafeSquare sq) {
    this->evaluator_->place_piece(cp, sq);
  }
  inline void decrement_piece_map(SafeColoredPiece cp, SafeSquare sq) {
    this->evaluator_->remove_piece(cp, sq);
  }

  inline std::string to_string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  void assert_valid_state() const;
  void assert_valid_state(const std::string& msg) const;

  // A checkmate on exactly the 100th half-move since a pawn move or capture will be considered drawn
  // here, so be careful about calling this in positions where there is a checkmate.
  bool is_draw_assuming_no_checkmate(unsigned plyFromRoot) const;
  bool is_draw_assuming_no_checkmate() const;

  bool is_3fold_repetition(unsigned plyFromRoot) const;
  bool is_fifty_move_rule() const;

  // This is used to avoid accidentally making a TT move in an invalid
  // position. It't unclear how necessary it is, since the only time we
  // blindly trust TT's bestMove is when checking for 3-fold draws (see
  // "search.h") and "illegal" moves are probably mostly okay here, since we
  // immediately undo them (e.g. capturing your own pieces or sliding over
  // pieces are fine).
  template<Color TURN>
  bool is_valid_move(Move move) const {
    // TODO: make more robust?
    return tiles_[move.from] != ColoredPiece::NO_COLORED_PIECE && cp2color(tiles_[move.to]) != TURN;
  }

 private:
  void _empty_();
};

#ifndef NDEBUG
extern Position *gDebugPos;
#endif

std::ostream& operator<<(std::ostream& stream, const Position& pos);

namespace {

// Maps (0 -> 0), (7 -> 1), (56 -> 2), and (63 -> 3)
uint8_t four_corners_to_byte(Bitboard b) {
  constexpr Bitboard mask = bb(SafeSquare::SA1) | bb(SafeSquare::SA8) | bb(SafeSquare::SH1) | bb(SafeSquare::SH8);
  return ((b & mask) * 0x1040000000000041) >> 60;
}

// Maps (4 -> {0, 1}) and (60 -> {3, 4})
uint8_t king_starts_to_byte(Bitboard b) {
  constexpr Bitboard mask = bb(SafeSquare::SE1) | bb(SafeSquare::SE8);
  constexpr Bitboard magic = bb(SafeSquare(6)) | bb(SafeSquare(7)) | bb(SafeSquare(60)) | bb(SafeSquare(61));
  return (((b & mask) >> 4) * magic) >> 60;
}

constexpr Bitboard kKingStartingPosition = bb(SafeSquare::SE1) | bb(SafeSquare::SE8);

}  // namespace

template<Color MOVER_TURN>
void undo(Position *pos) {
  pos->assert_valid_state();
  assert(pos->history_.size() > 0);
  assert(pos->turn_ == opposite_color<MOVER_TURN>());

  pos->turn_ = MOVER_TURN;

  const ExtMove extMove = pos->history_.back();
  pos->currentState_ = pos->states_.back();
  pos->history_.pop_back();
  pos->states_.pop_back();
  if (MOVER_TURN == Color::BLACK) {
    pos->wholeMoveCounter_ -= 1;
  }

  const Move move = extMove.move;
  const ColoredPiece movingPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<MOVER_TURN, Piece::PAWN>() : pos->tiles_[move.to];
  const ColoredPiece capturedPiece = extMove.capture;
  const ColoredPiece promoPiece = move.moveType == MoveType::PROMOTION ? coloredPiece<MOVER_TURN>(Piece(move.promotion + 2)) : movingPiece;
  const Location f = square2location(move.from);
  const Location t = square2location(move.to);
  const UnsafeSquare epSquare = pos->currentState_.epSquare;

  pos->pieceBitboards_[movingPiece] |= f;
  pos->pieceBitboards_[promoPiece] &= ~t;
  pos->colorBitboards_[MOVER_TURN] |= f;
  pos->colorBitboards_[MOVER_TURN] &= ~t;
  pos->tiles_[move.from] = movingPiece;
  pos->tiles_[move.to] = capturedPiece;

  pos->increment_piece_map(movingPiece, move.from);
  pos->decrement_piece_map(promoPiece, move.to);
  pos->increment_piece_map(capturedPiece, move.to);

  const bool hasCapturedPiece = (capturedPiece != ColoredPiece::NO_COLORED_PIECE);
  pos->pieceBitboards_[capturedPiece] |= t;
  pos->colorBitboards_[opposite_color<MOVER_TURN>()] |= t * hasCapturedPiece;

  if (move.moveType == MoveType::CASTLE) {
    if (MOVER_TURN == Color::BLACK) {
      assert(move.from == 4);
      assert(move.to == 2 || move.to == 6);
    } else {
      assert(move.from == 60);
      assert(move.to == 62 || move.to == 58);
    }
    SafeSquare rookDestination = SafeSquare((uint16_t(move.from) + uint16_t(move.to)) / 2);
    SafeSquare rookOrigin = SafeSquare(((uint16_t(move.to) % 8) * 7 - 14) / 4 + (MOVER_TURN == Color::WHITE ? 56 : 0));

    Bitboard rookDestinationBB = bb(rookDestination);
    Bitboard rookOriginBB = bb(rookOrigin);

    const ColoredPiece myRookPiece = coloredPiece<MOVER_TURN, Piece::ROOK>();
    pos->pieceBitboards_[myRookPiece] |= rookOriginBB;
    pos->pieceBitboards_[myRookPiece] &= ~rookDestinationBB;
    pos->colorBitboards_[MOVER_TURN] |= rookOriginBB;
    pos->colorBitboards_[MOVER_TURN] &= ~rookDestinationBB;
    pos->tiles_[rookDestination] = ColoredPiece::NO_COLORED_PIECE;
    pos->tiles_[rookOrigin] = myRookPiece;

    pos->increment_piece_map(myRookPiece, rookOrigin);
    pos->decrement_piece_map(myRookPiece, rookDestination);
  }

  if (UnsafeSquare(move.to) == epSquare && movingPiece == coloredPiece<MOVER_TURN, Piece::PAWN>()) {
    // TODO: get rid of if statement
    if (MOVER_TURN == Color::BLACK) {
      assert(move.from / 8 == 4);
      assert(move.to / 8 == 5);
    } else {
      assert(move.from / 8 == 3);
      assert(move.to / 8 == 2);
    }

    constexpr Color opposingColor = opposite_color<MOVER_TURN>();
    SafeSquare enpassantSq = SafeSquare((MOVER_TURN == Color::WHITE ? move.to + 8 : move.to - 8));
    Bitboard enpassantLocBB = bb(enpassantSq);

    constexpr ColoredPiece opposingPawn = coloredPiece<opposingColor, Piece::PAWN>();

    pos->pieceBitboards_[opposingPawn] |= enpassantLocBB;
    pos->colorBitboards_[opposingColor] |= enpassantLocBB;
    assert(pos->tiles_[enpassantSq] == ColoredPiece::NO_COLORED_PIECE);
    pos->tiles_[enpassantSq] = opposingPawn;

    // TODO: tell network about en passant square
    pos->increment_piece_map(opposingPawn, SafeSquare(enpassantSq));
  }
  pos->assert_valid_state();
}

template<Color TURN>
void foo(Position *pos) {
  pos->assert_valid_state();
}

template<Color TURN>
void bar(Position *pos) {
  pos->assert_valid_state();
}

}  // namespace ChessEngine

#endif  // POSITION_H