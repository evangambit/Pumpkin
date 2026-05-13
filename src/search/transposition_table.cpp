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
  this->resize(megabytes);
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

int score(const TTEntry& entry, uint8_t generation) {
  return (entry.bound == BoundType::EXACT) * 128 + (entry.generation == generation) * 256 + entry.depth;
}

void TranspositionTable::store(uint64_t key, Move bestMove, int8_t depth, Evaluation value, BoundType bound) {
  assert(depth >= std::numeric_limits<int8_t>::min() && depth <= std::numeric_limits<int8_t>::max());
  size_t idx = key % table_.size();
  spinLocks_[key % kNumSpinLocks].lock();
  TTEntry& entry = table_[idx];
  TTEntry newEntry = {key, bestMove, depth, value, bound, generation_};
  bool replace = score(newEntry, generation_) >= score(entry, generation_);
  if (key == entry.key && ((bound == BoundType::EXACT) == (entry.bound == BoundType::EXACT)) && depth < entry.depth) {
    replace = false;
  }
  if (replace) {
    entry = newEntry;
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
