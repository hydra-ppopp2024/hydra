/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

class ReduceOptions {
 public:
  using Func = std::function<void(void*, const void*, const void*, size_t)>;

  explicit ReduceOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->elements = buf->size / sizeof(T);//-》size获取所占空间的总字节数
    this->elementSize = sizeof(T);
    this->in = std::move(buf);//这里的实现和allreduce不大一样
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    this->elements = elements;
    this->elementSize = sizeof(T);
    this->in = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->elements = buf->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->out = std::move(buf);
  }

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    this->elements = elements;
    this->elementSize = sizeof(T);
    this->out = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  void setRoot(int root) {//?好像是找根节点
    this->root = root;
  }

  void setReduceFunction(Func fn) {
    this->reduce = fn;
  }

  void setTag(uint32_t tag) {
    this->tag = tag;
  }

  void setMaxSegmentSize(size_t maxSegmentSize) {
    this->maxSegmentSize = maxSegmentSize;
  }

  void setTimeout(std::chrono::milliseconds timeout) {//为什么都有timeout？
    this->timeout = timeout;
  }

 protected:
  std::shared_ptr<Context> context;
  std::unique_ptr<transport::UnboundBuffer> in;
  std::unique_ptr<transport::UnboundBuffer> out;

  // Number of elements.
  size_t elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Rank of process to reduce to.
  int root = -1;

  // Reduction function.
  Func reduce;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // This is the maximum size of each I/O operation (send/recv) of which
  // two are in flight at all times. A smaller value leads to more
  // overhead and a larger value leads to poor cache behavior.
  static constexpr size_t kMaxSegmentSize = 1024 * 1024;

  // Internal use only. This is used to exercise code paths where we
  // have more than 2 segments per rank without making the tests slow
  // (because they would require millions of elements if the default
  // were not configurable).
  size_t maxSegmentSize = kMaxSegmentSize;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  friend void reduce(ReduceOptions&);
};

void reduce(ReduceOptions& opts);

} // namespace gloo
