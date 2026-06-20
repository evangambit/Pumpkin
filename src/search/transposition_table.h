#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <unordered_map>


#include "../game/Position.h"
#include "../game/Move.h"
#include "../utils/SpinLock.h"

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
  int8_t depth;  // Negative values are used for depth into qsearch.
  Evaluation value;
  BoundType bound;
  uint8_t generation;
  TTEntry flip() const {
    TTEntry flipped = *this;
    flipped.value = -flipped.value;
    if (bound == BoundType::LOWER) {
      flipped.bound = BoundType::UPPER;
    } else if (bound == BoundType::UPPER) {
      flipped.bound = BoundType::LOWER;
    }
    return flipped;
  }
  Evaluation lowerbound() const {
    if (bound == BoundType::EXACT || bound == BoundType::LOWER) {
      return value;
    } else {
      return kMinEval;
    }
  }
  Evaluation upperbound() const {
    if (bound == BoundType::EXACT || bound == BoundType::UPPER) {
      return value;
    } else {
      return kMaxEval;
    }
  }
};
static_assert(sizeof(TTEntry) <= 16, "TTEntry should be 16 bytes or less to fit efficiently in cache lines");

constexpr size_t kNumSpinLocks = 2048;

class TranspositionTable {
 public:
  TranspositionTable(size_t megabytes);
  void clear();
  void new_search();
  void store(uint64_t key, Move bestMove, int8_t depth, Evaluation value, BoundType bound);
  bool probe(uint64_t key, TTEntry& entry);
  bool unsafe_probe(uint64_t key, TTEntry& entry) const;
  size_t kb_size() const { return table_.size() * sizeof(TTEntry) / 1024; }
  void resize(size_t megabytes) {
    const size_t bytes = megabytes * 1024 * 1024;
    size_t size = std::max<size_t>(1000, bytes / sizeof(TTEntry));  // Minimum size of 1000 entries.
    table_.resize(size);
    clear();
  }
  friend std::ostream& operator<<(std::ostream& os, const TranspositionTable& tt) {
    os << "TranspositionTable: " << tt.kb_size() << " KB, Entries: " << tt.table_.size();
    return os;
  }
private:
  std::vector<SpinLock> spinLocks_;
  std::vector<TTEntry> table_;
  uint8_t generation_ = 1;
};

} // namespace ChessEngine

#endif // TRANSPOSITION_TABLE_H
