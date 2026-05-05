#include "transposition_table.h"

#include <cassert>
#include <limits>
#include <cstring>

namespace ChessEngine {

std::string bound_type_to_string(BoundType bound) {
  switch (bound) {
    case BoundType::EXACT:
      return "EXACT";
    case BoundType::LOWER:
      return "LOWER";
    case BoundType::UPPER:
      return "UPPER";
    default:
      return "UNKNOWN";
  }
}

TranspositionTable::TranspositionTable(size_t megabytes) {
  table_.resize(1024 * 1024 * megabytes);
  spinLocks_.resize(kNumSpinLocks);
}

void TranspositionTable::new_search() {
  ++generation_;
  // Avoid 0 as valid generation. This way we can identify empty entries
  // without needing an extra flag or relying on hashes to never be 0.
  if (generation_ == 0) {
    generation_ = 1;
  }
}

void TranspositionTable::clear() {
  std::memset(table_.data(), 0, sizeof(TTEntry) * table_.size());
}

void TranspositionTable::store(uint64_t key, Move bestMove, int depth, int value, BoundType bound) {
  assert(depth >= std::numeric_limits<int8_t>::min() && depth <= std::numeric_limits<int8_t>::max());
  size_t idx = key % table_.size();
  spinLocks_[key % kNumSpinLocks].lock();
  TTEntry& entry = table_[idx];
  bool replace = false;
  if (entry.generation != generation_) {
    replace = true;
  } else if (bound == BoundType::EXACT && entry.bound != BoundType::EXACT) {
    replace = true;
  } else if (bound == BoundType::EXACT && entry.bound == BoundType::EXACT && depth >= entry.depth) {
    replace = true;
  } else if (bound != BoundType::EXACT && entry.bound != BoundType::EXACT && depth >= entry.depth) {
    replace = true;
  }
  if (key == entry.key && ((bound == BoundType::EXACT) == (entry.bound == BoundType::EXACT)) && depth < entry.depth) {
    replace = false;
  }
  if (replace) {
    entry.key = key;
    entry.bestMove = bestMove;
    entry.depth = depth;
    entry.value = value;
    entry.bound = bound;
    entry.generation = generation_;
  }
  spinLocks_[key % kNumSpinLocks].unlock();
}

bool TranspositionTable::probe(uint64_t key, TTEntry& entry) {
  size_t idx = key % table_.size();
  spinLocks_[key % kNumSpinLocks].lock();
  const TTEntry& found = table_[idx];
  if (found.key == key) {
    entry = found;
    spinLocks_[key % kNumSpinLocks].unlock();
    return true;
  }
  spinLocks_[key % kNumSpinLocks].unlock();
  return false;
}

bool TranspositionTable::unsafe_probe(uint64_t key, TTEntry& entry) const {
  size_t idx = key % table_.size();
  const TTEntry& found = table_[idx];
  if (found.key == key) {
    entry = found;
    return true;
  }
  return false;
}

} // namespace ChessEngine
