/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gloo/transport/pair.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {
namespace transport {

// The context represents a set of pairs that belong to the same
// group. It is roughly equivalent to the top level context class
// with the exception that it captures transport specifics.
//
// While implementing the recv-from-any functionality we realized we
// realized we needed some transport-specific state shared between all
// pairs in a group, to arbitrate between multiple pairs attempting to
// send to the same buffer.
//
class Context {
 public:
  using slot_t = uint64_t;
  using rank_t = int;

  Context(int rank, int size);

  virtual ~Context();//析构函数

  const int rank;
  const int size;

  virtual std::unique_ptr<Pair>& getPair(int rank);//虚函数，联想猫狗 speak 例子来解释

  virtual std::unique_ptr<Pair>& createPair(int rank) = 0;

  // Creates unbound buffer to be used with the ranks in this context.
  // It is not bound to a specific rank, but still bound to this
  // context. This is needed to support recv-from-any semantics, where
  // the context is used as shared arbiter between pairs that are
  // ready to send and buffers that are ready to receive.
  virtual std::unique_ptr<transport::UnboundBuffer> createUnboundBuffer(
      void* ptr,
      size_t size) = 0;

  void setTimeout(std::chrono::milliseconds timeout) {
    timeout_ = timeout;
  }

  std::chrono::milliseconds getTimeout() const {
    return timeout_;
  }

 protected:
  // Protects access to the pending operations and expected
  // notifications vectors. These vectors can only be mutated by an
  // instance of the Context::Mutator class, which acquires this lock
  // upon construction.
  //
  // The vector of pairs is logically const and may be accessed
  // without holding this lock.
  //
  // If this lock is acquired from a function on a Pair class, ensure
  // that Pair's instance lock is acquired before acquiring this lock.
  //
  std::mutex mutex_;//锁

  // Lifecycle of the pairs is managed by a std::unique_ptr of the
  // base class. This is done because the public context API dictates
  // that getPair() returns a reference to this type. Functions
  // internal to this class can cast these points to the native
  // transport specific type.
  std::vector<std::unique_ptr<Pair>> pairs_;//

  // Default timeout for new pairs (e.g. during initialization) and
  // any kind of send/recv operation.
  std::chrono::milliseconds timeout_;//超时时间

 protected:
  // Keep track of pending send and recv notifications or operations
  // for a single slot.
  //
  // The order with which ranks register a pending operation is
  // preserved to avoid starvation for recv-from-any operations.
  //
  class Tally final {
   private:
    // This class stores the ranks with pending operations against
    // this slot. The parent class uses two instances: one for send
    // operations and one for receive operations.
    //
    // Adding ranks with a pending operation is done through the
    // `push` method, which adds the rank to the end of the list.
    // Removing ranks with a pending operation is done through the
    // `shift` method, which removes a specific rank from the
    // beginning of the list. The caller must know the rank to remove
    // prior to calling `shift`, because a receive operation may be
    // limited to a subset of ranks.
    //
    class List final {//跟踪排名列表的私有子类。
     public:
      bool empty() const {
        return ranks_.empty();//函数返回该 Tally 实例是否还有待处理的操作。
      }

      const std::vector<rank_t>& list() const {
        return ranks_;
      }

      // Push rank to the end of the list.
      void push(rank_t rank) {
        ranks_.push_back(rank);
      }

      // Shift rank from the beginning of the list.
      // Returns if the rank could be found and was removed.
      bool shift(rank_t rank) {
        auto it = std::find(ranks_.begin(), ranks_.end(), rank);
        if (it != ranks_.end()) {
          ranks_.erase(it);
          return true;
        }
        return false;
      }

     private:
      std::vector<rank_t> ranks_;
    };

   public:
    explicit Tally(slot_t slot) : slot(slot) {}

    slot_t slot;

    bool empty() const {
      return send_.empty() && recv_.empty();
    }

    const std::vector<rank_t>& getSendList() const {//返回待处理的 send 和 recv 操作关联的排名列表。
      return send_.list();
    }

    const std::vector<rank_t>& getRecvList() const {
      return recv_.list();
    }

    void pushSend(rank_t rank) {//用于在 Tally 实例中添加指定排名的 send 或 recv 操作。
      send_.push(rank);
    }

    void pushRecv(rank_t rank) {
      recv_.push(rank);
    }

    bool shiftSend(rank_t rank) {//于将指定的排名从 Tally 实例的 send 或 recv 操作列表中移除
      return send_.shift(rank);
    }

    bool shiftRecv(rank_t rank) {
      return recv_.shift(rank);
    }

   private:
    List send_;
    List recv_;
  };

  // This class is used to locate a tally for a specific slot in a
  // vector container. If the tally if needed and doesn't exist, it
  // may be lazily created. If, on destruction, the tally that this
  // class points to is empty, it is removed from the container.
  //
  // This functionality is needed both for the pending operation tally
  // and for the expected notification tally. Therefore, we chose to
  // use a somewhat generalized class, over duplicating the same
  // functionality for two identical containers.
  //
  class LazyTally final {
   public:
    LazyTally(std::vector<Tally>& vec, slot_t slot);

    ~LazyTally();

    // Returns if a Tally instance exists for the specified slot.返回该槽位的 Tally 实例是否存在。
    bool exists();

    // Returns pointer to Tally instance for the specified slot.
    // Lazily constructs a new instance if needed.返回该槽位的 Tally 实例，并在需要时创建该实例
    Tally& get();

   private:
    // Reference to underlying container.
    std::vector<Tally>& vec_;

    // Slot for the tally we're interested in.
    const slot_t slot_;

    // Iterator to tally for the specified slot.
    std::vector<Tally>::iterator it_;

    // If the iterator has been initialized.
    bool initialized_;

    // Initialize iterator to Tally instance for this slot.
    void initialize_iterator();
  };

  // This class is used to mutate the pending operation tally and
  // expected notification tally for a specific source rank against a
  // specific slot. An instance is expected to have a short lifetime
  // as it holds a lock on the parent context object.
  class Mutator final {
   public:
    Mutator(Context& context, slot_t slot, rank_t rank);

    void pushRemotePendingRecv();

    void pushRemotePendingSend();

    bool shiftRemotePendingRecv();

    bool shiftRemotePendingSend();

    // When posting a receive operation, we first check if a send
    // notification for the specified slot was already received. If
    // not, it may already been in flight, and we must take care to
    // ignore it when it arrives. This function ensures that the next
    // send notification for this slot is ignored.
    //
    // Also see `shiftExpectedSendNotification`. 用于管理在指定槽位上等待 send 通知的状态。
    //
    void pushExpectedSendNotification();

    // This function returns whether or not we were expecting a send
    // notification for this slot. If we do, we can ignore it.
    //
    // Also see `pushExpectedSendNotification`.
    //
    bool shiftExpectedSendNotification();

   private:
    std::lock_guard<std::mutex> lock_;
    Context& context_;
    const slot_t slot_;
    const rank_t rank_;

    // Find and mutate pending operation tally.
    LazyTally pendingOperations_;

    // Find and mutate expected notification tally.
    LazyTally expectedNotifications_;
  };

  // The pending operation tally is stored as a vector under the
  // assumption that we're working with very few of them. It should be
  // cheaper to perform a linear search in contiguous memory than it
  // is to maintain a map of them and pay a higher mutation overhead.
  std::vector<Tally> pendingOperations_;

  // If a recv operation is posted before the corresponding send
  // notification is received, then we need to make sure the send
  // notification isn't added to the pending operations vector. To do
  // so, we maintain a structure of notifications we expect to
  // receive, so that they can be dropped when they are.
  std::vector<Tally> expectedNotifications_;

  // Permit the mutator class to touch the pending operation tally.
  friend class Mutator;

 protected:
  // Return iterator to pending operation tally for specific slot.
  std::vector<Tally>::iterator findPendingOperations(slot_t slot);
};

} // namespace transport
} // namespace gloo
