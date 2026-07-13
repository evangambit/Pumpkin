#ifndef SRC_SPINLOCK_H
#define SRC_SPINLOCK_H

#include <atomic>

namespace ChessEngine {

// Fast spinlock for short critical sections (TT, search manager).
// Do not hold across I/O, large memsets, or other slow work — use std::mutex for those.
struct SpinLock {
  std::atomic<bool> lock_ = {false};
  SpinLock() = default;
  SpinLock(SpinLock&&) noexcept : lock_{false} {}
  SpinLock& operator=(SpinLock&&) = delete;
  SpinLock(const SpinLock&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;

  void lock() {
    while (lock_.exchange(true, std::memory_order_acquire)) {}
  }

  void unlock() { lock_.store(false, std::memory_order_release); }
};

}  // namespace ChessEngine

#endif  // SRC_SPINLOCK_H
