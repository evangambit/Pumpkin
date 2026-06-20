#ifndef SRC_SPINLOCK_H
#define SRC_SPINLOCK_H

#include <atomic>

namespace ChessEngine {

struct SpinLock {
  std::atomic<bool> lock_ = {false};
  SpinLock() = default;
  SpinLock(SpinLock&&) noexcept : lock_{false} {}
  SpinLock& operator=(SpinLock&&) = delete;
  SpinLock(const SpinLock&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;
  void lock() { while(lock_.exchange(true)); }
  void unlock() { lock_.store(false); }
};

}  // namespace ChessEngine

#endif  // SRC_SPINLOCK_H