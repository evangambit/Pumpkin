#ifndef BOARDLISTENER_H
#define BOARDLISTENER_H

#include <cassert>
#include <cstdint>
#include <cstring>

#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "geometry.h"
#include "utils.h"
#include "Move.h"
#include "../string_utils.h"

namespace ChessEngine {

struct BoardListener {
    virtual ~BoardListener() = default;
    virtual void empty() = 0;
    virtual void place_piece(ColoredPiece cp, SafeSquare square) = 0;
    virtual void remove_piece(ColoredPiece cp, SafeSquare square) = 0;
    virtual void place_piece(SafeColoredPiece cp, SafeSquare square) = 0;
    virtual void remove_piece(SafeColoredPiece cp, SafeSquare square) = 0;
};

struct DummyBoardListener : public BoardListener {
    void empty() override {}
    void place_piece(ColoredPiece cp, SafeSquare square) override {}
    void remove_piece(ColoredPiece cp, SafeSquare square) override {}
    void place_piece(SafeColoredPiece cp, SafeSquare square) override {}
    void remove_piece(SafeColoredPiece cp, SafeSquare square) override {}
};

}  // namespace ChessEngine

#endif  // BOARDLISTENER_H
