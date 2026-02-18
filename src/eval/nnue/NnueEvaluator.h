#ifndef SRC_EVAL_NNUE_NNUEEVALUATIOR_H
#define SRC_EVAL_NNUE_NNUEEVALUATIOR_H

#include <chrono>
#include <iostream>
#include <fstream>

#include "Nnue.h"
#include "Utils.h"

#include "../../game/Position.h"
#include "../Evaluator.h"

using namespace ChessEngine;

namespace NNUE {

struct WDL {
  double win;
  double draw;
  double loss;
};

struct NnueEvaluator : public EvaluatorInterface {
  std::shared_ptr<Nnue> nnue_model;

  Bitboard lastPieceBitboards[NF_COUNT];

  NnueEvaluator(std::shared_ptr<Nnue> model) : nnue_model(model) {
    this->empty();
  }

  // Board listener
  void empty() override {
    nnue_model->clear_accumulator();
    std::fill_n(lastPieceBitboards, NF_COUNT, kEmptyBitboard);
  }
  void place_piece(ColoredPiece cp, SafeSquare square) override {
  }
  void remove_piece(ColoredPiece cp, SafeSquare square) override {
  }
  void place_piece(SafeColoredPiece cp, SafeSquare square) override {
  }
  void remove_piece(SafeColoredPiece cp, SafeSquare square) override {
  }


  // EvaluatorInterface

  ColoredEvaluation<Color::WHITE> evaluate_white(const Position& pos) override {
    assert(pos.turn_ == Color::WHITE);
    Evaluation eval = _evaluate(pos);
    return ColoredEvaluation<Color::WHITE>(eval);
  }
  ColoredEvaluation<Color::BLACK> evaluate_black(const Position& pos) override {
    assert(pos.turn_ == Color::BLACK);
    Evaluation eval = _evaluate(pos);
    return ColoredEvaluation<Color::BLACK>(eval);
  }

  Evaluation _evaluate(const Position& pos) {
    if ((pos.pieceBitboards_[ColoredPiece::WHITE_PAWN] | pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]) == kEmptyBitboard) {
      int whiteMinor = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP]);
      int blackMinor = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT] | pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP]);
      int whiteMajor = std::popcount(pos.pieceBitboards_[ColoredPiece::WHITE_ROOK] | pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN]);
      int blackMajor = std::popcount(pos.pieceBitboards_[ColoredPiece::BLACK_ROOK] | pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN]);
      if (whiteMajor + blackMajor == 0 && whiteMinor <= 1 && blackMinor <= 1) {
        // No pawns, no major pieces, and <= 1 minor. Almost certainly a draw.
        return Evaluation(0);
      }
    }
    #ifndef NDEBUG
      Vector<1024> accCopy = nnue_model->whiteAcc;
    #endif

    for (NnueFeatureBitmapType i = static_cast<NnueFeatureBitmapType>(0); i < NF_COUNT; i = static_cast<NnueFeatureBitmapType>(i + 1)) {
      const Bitboard oldBitboard = lastPieceBitboards[i];
      Bitboard newBitboard;
      switch (i) {
        case NF_WHITE_PAWN:
          newBitboard = pos.pieceBitboards_[ColoredPiece::WHITE_PAWN];
          break;
        case NF_WHITE_KNIGHT:
          newBitboard = pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT];
          break;
        case NF_WHITE_BISHOP:
          newBitboard = pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP];
          break;
        case NF_WHITE_ROOK:
          newBitboard = pos.pieceBitboards_[ColoredPiece::WHITE_ROOK];
          break;
        case NF_WHITE_QUEEN:
          newBitboard = pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN];
          break;
        case NF_WHITE_KING:
          newBitboard = pos.pieceBitboards_[ColoredPiece::WHITE_KING];
          break;
        case NF_BLACK_PAWN:
          newBitboard = pos.pieceBitboards_[ColoredPiece::BLACK_PAWN];
          break;
        case NF_BLACK_KNIGHT:
          newBitboard = pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT];
          break;
        case NF_BLACK_BISHOP:
          newBitboard = pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP];
          break;
        case NF_BLACK_ROOK:
          newBitboard = pos.pieceBitboards_[ColoredPiece::BLACK_ROOK];
          break;
        case NF_BLACK_QUEEN:
          newBitboard = pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN];
          break;
        case NF_BLACK_KING:
          newBitboard = pos.pieceBitboards_[ColoredPiece::BLACK_KING];
          break;
        case NF_HANGING_PIECES:
          Threats threats;
          create_threats(pos.pieceBitboards_, pos.colorBitboards_, &threats);
          newBitboard = threats.badForCp(ColoredPiece::WHITE_PAWN) & pos.pieceBitboards_[ColoredPiece::WHITE_PAWN];
          newBitboard |= threats.badForCp(ColoredPiece::WHITE_KNIGHT) & pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT];
          newBitboard |= threats.badForCp(ColoredPiece::WHITE_BISHOP) & pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP];
          newBitboard |= threats.badForCp(ColoredPiece::WHITE_ROOK) & pos.pieceBitboards_[ColoredPiece::WHITE_ROOK];
          newBitboard |= threats.badForCp(ColoredPiece::WHITE_QUEEN) & pos.pieceBitboards_[ColoredPiece::WHITE_QUEEN];
          newBitboard |= threats.badForCp(ColoredPiece::WHITE_KING) & pos.pieceBitboards_[ColoredPiece::WHITE_KING];
          newBitboard |= threats.badForCp(ColoredPiece::BLACK_PAWN) & pos.pieceBitboards_[ColoredPiece::BLACK_PAWN];
          newBitboard |= threats.badForCp(ColoredPiece::BLACK_KNIGHT) & pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT];
          newBitboard |= threats.badForCp(ColoredPiece::BLACK_BISHOP) & pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP];
          newBitboard |= threats.badForCp(ColoredPiece::BLACK_ROOK) & pos.pieceBitboards_[ColoredPiece::BLACK_ROOK];
          newBitboard |= threats.badForCp(ColoredPiece::BLACK_QUEEN) & pos.pieceBitboards_[ColoredPiece::BLACK_QUEEN];
          newBitboard |= threats.badForCp(ColoredPiece::BLACK_KING) & pos.pieceBitboards_[ColoredPiece::BLACK_KING];
          break;
        default:
          std::cerr << "Invalid NnueFeatureBitmapType: " << i << std::endl;
      }
      Bitboard diff = oldBitboard & ~newBitboard;
      while (diff) {
        const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(diff);
        nnue_model->decrement(feature_index(i, sq));
      }
      diff = ~oldBitboard & newBitboard;
      while (diff) {
        const SafeSquare sq = pop_lsb_i_promise_board_is_not_empty(diff);
        nnue_model->increment(feature_index(i, sq));
      }
      lastPieceBitboards[i] = newBitboard;
    }

    if (((pos.currentState_.castlingRights & kCastlingRights_WhiteKing) > 0) != nnue_model->x[WHITE_KINGSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_WhiteKing)) {
        nnue_model->increment(WHITE_KINGSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(WHITE_KINGSIDE_CASTLING_RIGHT);
      }
    }
    if (((pos.currentState_.castlingRights & kCastlingRights_WhiteQueen) > 0) != nnue_model->x[WHITE_QUEENSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_WhiteQueen)) {
        nnue_model->increment(WHITE_QUEENSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(WHITE_QUEENSIDE_CASTLING_RIGHT);
      }
    }
    if (((pos.currentState_.castlingRights & kCastlingRights_BlackKing) > 0) != nnue_model->x[BLACK_KINGSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_BlackKing)) {
        nnue_model->increment(BLACK_KINGSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(BLACK_KINGSIDE_CASTLING_RIGHT);
      }
    }
    if (((pos.currentState_.castlingRights & kCastlingRights_BlackQueen) > 0) != nnue_model->x[BLACK_QUEENSIDE_CASTLING_RIGHT]) {
      if ((pos.currentState_.castlingRights & kCastlingRights_BlackQueen)) {
        nnue_model->increment(BLACK_QUEENSIDE_CASTLING_RIGHT);
      } else {
        nnue_model->decrement(BLACK_QUEENSIDE_CASTLING_RIGHT);
      }
    }

    int16_t *eval = nnue_model->forward(pos.turn_);
    int16_t score = eval[0];

    #ifndef NDEBUG
      nnue_model->compute_acc_from_scratch(pos);
      int16_t score2 = nnue_model->forward(pos.turn_)[0];
      if (score != score2) {
        std::cerr << "NNUE evaluation mismatch: " << score << " vs " << score2 << std::endl;
        accCopy.print_diff(nnue_model->whiteAcc);
        throw std::runtime_error("NNUE evaluation mismatch");
      }
    #endif

    return Evaluation(score);
  }

  std::shared_ptr<EvaluatorInterface> clone() const override {
    return std::make_shared<NnueEvaluator>(this->nnue_model->clone());
  }
  std::string to_string() const override {
    return "NnueEvaluator";
  }
};

} // namespace NNUE

#endif  // SRC_EVAL_NNUE_NNUEEVALUATIOR_H
