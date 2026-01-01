#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <cstdint>
#include <unordered_map>
#include "../game/Position.h"
#include "../game/Move.h"

namespace ChessEngine {

enum class BoundType : uint8_t {
  EXACT,
  LOWER,
  UPPER
};

struct TTEntry {
  uint64_t key;
  Move bestMove;
  int depth;
  int value;
  BoundType bound;
  int ply;
  uint32_t generation;
};

class TranspositionTable {
public:
  TranspositionTable(size_t kilobytes);
  void clear();
  void new_search();
  void store(uint64_t key, Move bestMove, int depth, int value, BoundType bound, int ply);
  bool probe(uint64_t key, TTEntry& entry) const;
  size_t kb_size() const { return size_ * sizeof(TTEntry) / 1024; }
  void resize(size_t kilobytes) {
    size_ = std::max<size_t>(1, (kilobytes * 1024) / sizeof(TTEntry));
    table_.resize(size_);
    clear();
  }
  friend std::ostream& operator<<(std::ostream& os, const TranspositionTable& tt) {
    os << "TranspositionTable: " << tt.kb_size() << " KB, Entries: " << tt.size_;
    return os;
  }
private:
  std::vector<TTEntry> table_;
  size_t size_;
  uint32_t generation_ = 1;
};

} // namespace ChessEngine

#endif // TRANSPOSITION_TABLE_H
