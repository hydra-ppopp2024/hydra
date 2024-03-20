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
#include <vector>
#include <iostream>
#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace gloo {

namespace detail {

struct AllreduceOptionsImpl {
  // This type describes the function to use for element wise reduction.
  //
  // Its arguments are:
  //   1. non-const output pointer
  //   2. const input pointer 1 (may be equal to 1)
  //   3. const input pointer 2 (may be equal to 1)
  //   4. number of elements to reduce.
  //
  // Note that this function is not strictly typed and takes void pointers.
  // This is specifically done to avoid the need for a templated options class
  // and templated algorithm implementations. We found this adds very little
  // value for the increase in compilation time and code size.
  //
  using Func = std::function<void(void*, const void*, const void*, size_t)>;

  enum Algorithm {
    UNSPECIFIED = 0,
    RING = 1,
    BCUBE = 2,
  };

  explicit AllreduceOptionsImpl(const std::shared_ptr<Context>& context)//初始化，相当于对context、timeout、al三个变量的初始化
      : context(context),
        timeout(context->getTimeout()),
        algorithm(UNSPECIFIED) {}

  std::shared_ptr<Context> context;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  // Algorithm selection.
  Algorithm algorithm;

  // Input and output buffers.
  // The output is used as input if input is not specified.
  std::vector<std::unique_ptr<transport::UnboundBuffer>> in;
  std::vector<std::unique_ptr<transport::UnboundBuffer>> out;

  // Number of elements.
  size_t elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

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
};

} // namespace detail

class AllreduceOptions {
 public:
  using Func = detail::AllreduceOptionsImpl::Func;
  using Algorithm = detail::AllreduceOptionsImpl::Algorithm;

  explicit AllreduceOptions(const std::shared_ptr<Context>& context)//这是个构造函数
      : impl_(context) {}//将参数 context 直接传递给 impl_ 成员的构造函数 AllreduceOptionsImpl(const std::shared_ptr<Context>& context) 进行构造。

  void setAlgorithm(Algorithm algorithm) {
    impl_.algorithm = algorithm;
  }

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
        std::cout << "elements111 " << std::endl;   
    std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs(1);
    bufs[0] = std::move(buf);
    setInputs<T>(std::move(bufs));
  }

  template <typename T>
  void setInputs(std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs) {
    impl_.elements = bufs[0]->size / sizeof(T);
    impl_.elementSize = sizeof(T);
    impl_.in = std::move(bufs);
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
      // std::cout << "ptrin " << ptr << std::endl;
    setInputs(&ptr, 1, elements);
  }

  template <typename T>
  void setInputs(std::vector<T*> ptrs, size_t elements) {
    setInputs(ptrs.data(), ptrs.size(), elements);//指针的首地址，长度
  }

  template <typename T>
  void setInputs(T** ptrs, size_t len, size_t elements) {//len的大小与节点数相等
    impl_.elements = elements;
    impl_.elementSize = sizeof(T);
    impl_.in.reserve(len);
    for (size_t i = 0; i < len; i++) {
      impl_.in.push_back(//pushback是装入的意思，这里面一石二鸟，输入的变量就是ptrs数组所指向的指针地址，注意**ptrs
          impl_.context->createUnboundBuffer(ptrs[i], elements * sizeof(T)));//前者是要装入的数据，后者是要申请的总空间
    }
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs(1);
    bufs[0] = std::move(buf);
    setOutputs<T>(std::move(bufs));
  }

  template <typename T>
  void setOutputs(std::vector<std::unique_ptr<transport::UnboundBuffer>> bufs) {
    impl_.elements = bufs[0]->size / sizeof(T);
    impl_.elementSize = sizeof(T);
    impl_.out = std::move(bufs);
  }
  //yed add
  detail::AllreduceOptionsImpl& getImpl() {
  return impl_;
}


  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    // std::cout << "ptrout " << ptr << std::endl;
    setOutputs(&ptr, 1, elements);
  }

  template <typename T>
  void setOutputs(std::vector<T*> ptrs, size_t elements) {
    setOutputs(ptrs.data(), ptrs.size(), elements);
  }

  template <typename T>
  void setOutputs(T** ptrs, size_t len, size_t elements) {
    impl_.elements = elements;
    impl_.elementSize = sizeof(T);
    impl_.out.reserve(len);
    for (size_t i = 0; i < len; i++) {
      impl_.out.push_back(
          impl_.context->createUnboundBuffer(ptrs[i], elements * sizeof(T)));
    }
  }

  void setReduceFunction(Func fn) {
    impl_.reduce = fn;
  }

  void setTag(uint32_t tag) {
    impl_.tag = tag;
  }

  void setMaxSegmentSize(size_t maxSegmentSize) {
    impl_.maxSegmentSize = maxSegmentSize;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    impl_.timeout = timeout;
  }

 protected:
  detail::AllreduceOptionsImpl impl_;

  friend void allreduce(const AllreduceOptions&);//括号内写明是谁的友元函数
};

void allreduce(const AllreduceOptions& opts);

} // namespace gloo
