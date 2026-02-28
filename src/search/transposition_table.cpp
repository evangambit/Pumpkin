#include "transposition_table.h"
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

TranspositionTable::TranspositionTable(size_t kilobytes) {
  resize(kilobytes);
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
  size_t idx = key % table_.size();
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
}

bool TranspositionTable::probe(uint64_t key, TTEntry& entry) const {
  size_t idx = key % table_.size();
  const TTEntry& found = table_[idx];
  if (found.key == key) {
    entry = found;
    return true;
  }
  return false;
}

} // namespace ChessEngine
