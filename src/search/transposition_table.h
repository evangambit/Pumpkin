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

std::string bound_type_to_string(BoundType bound);

struct TTEntry {
  uint64_t key;
  Move bestMove;
  uint8_t depth;
  Evaluation value;
  BoundType bound;
  uint8_t generation;
};

class TranspositionTable {
 public:
  TranspositionTable(size_t kilobytes);
  void clear();
  void new_search();
  void store(uint64_t key, Move bestMove, int depth, int value, BoundType bound);
  bool probe(uint64_t key, TTEntry& entry) const;
  size_t kb_size() const { return table_.size() * sizeof(TTEntry) / 1024; }
  void resize(size_t kilobytes) {
    size_t size = std::max(1000LU, (kilobytes * 1024) / sizeof(TTEntry));  // Minimum size of 1000 entries.
    table_.resize(size);
    clear();
  }
  friend std::ostream& operator<<(std::ostream& os, const TranspositionTable& tt) {
    os << "TranspositionTable: " << tt.kb_size() << " KB, Entries: " << tt.table_.size();
    return os;
  }
private:
  std::vector<TTEntry> table_;
  uint8_t generation_ = 1;
};

} // namespace ChessEngine

#endif // TRANSPOSITION_TABLE_H
