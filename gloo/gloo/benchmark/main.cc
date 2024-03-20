/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include<time.h>
#include <ctime>
#include <random>
#include <sys/mman.h>
#include <sys/shm.h>
#include <future>
#include <thread>
#include "gloo/allgather_ring.h"
#include "gloo/allreduce.h"
#include "gloo/sharp_allreduce.h"
#include "gloo/sharp_reduce.h"
#include "gloo/sharp_bcast.h"
#include "gloo/sharp_allgather.h"
#include "gloo/sharp_barrier.h"
#include "gloo/pipeallreduce.h"
#include "gloo/pipeallreduce-a.h"
#include "gloo/pipeallreduce-s.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/pairwise_exchange.h"
#include "gloo/reduce_scatter.h"
#include "gloo/reduce.h"
#include "gloo/common/aligned_allocator.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include "gloo/types.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/newbenchmark.h"
#include "gloo/benchmark/runner.h"






using namespace gloo;
using namespace gloo::benchmark;

namespace {

template <typename T>
class AllgatherBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto inPtrs = this->allocate(this->options_.inputs, elements);
    GLOO_ENFORCE_EQ(inPtrs.size(), this->options_.inputs);
    outputs_.resize(this->options_.inputs * this->context_->size * elements);
    this->algorithm_.reset(new AllgatherRing<T>(
        this->context_, this->getInputs(), outputs_.data(), elements));
  }

  void verify() override {
    const auto stride = this->context_->size * this->inputs_.size();
    const auto elements = this->inputs_[0].size();
    for (int rank = 0; rank < this->context_->size; rank++) {
      auto val = rank * this->inputs_.size();
      for (int elem = 0; elem < elements; elem++) {
        T exp(elem * stride + val);
        for (int input = 0; input < this->inputs_.size(); input++) {
          const auto rankOffset = rank * elements * this->inputs_.size();
          const auto inputOffset = input * elements;
          GLOO_ENFORCE_EQ(
            outputs_[rankOffset + inputOffset + elem], exp + T(input),
            "Mismatch at index: [", rank, ", ", input, ", ", elem, "]");
        }
      }
    }
  }

 protected:
  std::vector<T> outputs_;
};

template <class A, typename T>
class AllreduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(new A(this->context_, ptrs, elements));
  }

  void verify() override {
    // Size is the total number of pointers across the context
    const auto size = this->context_->size * this->inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // allreduce, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : this->inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(offset + expected), input[i], "Mismatch at index: ", i);
      }
    }
  }
};

template <typename T>
class BarrierAllToAllBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t /* unused */) override {
    this->algorithm_.reset(new BarrierAllToAll(this->context_));
  }
};

template <typename T>
class BarrierAllToOneBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t /* unused */) override {
    // This tool measures at rank=0, so use root=1 for the all to one
    // barrier to measure the end-to-end latency (otherwise we might
    // not account for the send-to-root part of the algorithm).
    this->algorithm_.reset(new BarrierAllToOne(this->context_, 1));
  }
};

template <typename T>
class BroadcastOneToAllBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    this->algorithm_.reset(
        new BroadcastOneToAll<T>(this->context_, ptrs, elements, rootRank_));
  }

  void verify() override {
    const auto stride = this->context_->size * this->inputs_.size();
    for (const auto& input : this->inputs_) {
      for (int i = 0; i < input.size(); i++) {
        auto offset = i * stride;
        GLOO_ENFORCE_EQ(
            T(offset + rootRank_), input[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
  const int rootRank_ = 0;
};

template <typename T>
class ReduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;

  public:
    ReduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
      int *inputPointers = reinterpret_cast<int*>(malloc(sizeof(int) * elements));
      int *outputPointers = reinterpret_cast<int*>(malloc(sizeof(int) * elements));
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements);
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1);
        outputPointers[i] = 0;
      }
      // auto inPtrs = this->allocate(this->options_.inputs, elements);
      // output_.resize(elements);

      // Configure ReduceOptions struct
      // Use rank 0 as root
      opts_.setRoot(rootRank_);
      // Set reduce function
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts_.setReduceFunction(fn);
      // MaxSegmentSize must be a multiple of the element size T
      // Can't be too small otherwise benchmark will run for a long time
      // Use a factor of (elements / 2)
      // opts_.setMaxSegmentSize(sizeof(T) * elements * 2);
      

    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options
      reduce(opts_);
    }

    void verify() override {
      // Size is the total number of pointers across the context
      // const auto size = this->context_->size * this->inputs_.size();
      // // Expected is set to be the expected value of ptr[0]
      // // after reduce gets called (calculation depends on the
      // // reduction function used and how we initialized the inputs)
      // const auto expected = (size * (size - 1)) / 2;
      // // The stride between values at subsequent indices is equal to
      // // "size", and we have "size" of them. Therefore, after
      // // reduce, the stride between expected values is "size^2".
      // const auto stride = size * size;

      // // Verify only for root
      // if (this->context_->rank == rootRank_) {
      //   for (int i = 0; i < output_.size(); i++) {
      //     auto offset = i * stride;
      //     GLOO_ENFORCE_EQ(
      //         T(offset + expected), output_[i],
      //         "Mismatch at index: ", i);
      //   }
      // }
    }

  protected:
    ReduceOptions opts_;

    // Always use rank 0 as the root
    const int rootRank_ = 0;
    // std::vector<T> output_;
};

template <typename T>
class PairwiseExchangeBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    this->algorithm_.reset(new PairwiseExchange(
        this->context_, elements, this->getOptions().destinations));
  }
};

template <typename T>
class ReduceScatterBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
 public:
  void initialize(size_t elements) override {
    auto ptrs = this->allocate(this->options_.inputs, elements);
    auto rem = elements;
    auto chunkSize =
        (elements + this->context_->size - 1) / this->context_->size;
    for (int i = 0; i < this->context_->size; ++i) {
      recvCounts_.push_back(std::min(chunkSize, rem));
      rem = rem > chunkSize ? rem - chunkSize : 0;
    }
    this->algorithm_.reset(
        new ReduceScatterHalvingDoubling<T>(
            this->context_, ptrs, elements, recvCounts_));
  }

  void verify() override {
    // Size is the total number of pointers across the context
    const auto size = this->context_->size * this->inputs_.size();
    // Expected is set to the expected value at ptr[0]
    const auto expected = (size * (size - 1)) / 2;
    // The stride between values at subsequent indices is equal to
    // "size", and we have "size" of them. Therefore, after
    // reduce-scatter, the stride between expected values is "size^2".
    const auto stride = size * size;
    for (const auto& input : this->inputs_) {
      int numElemsSoFar = 0;
      for (int i = 0; i < this->context_->rank; ++i) {
          numElemsSoFar += recvCounts_[i];
      }
      for (int i = 0; i < recvCounts_[this->context_->rank]; ++i) {
        auto offset = (numElemsSoFar + i) * stride;
        GLOO_ENFORCE_EQ(
            T(offset + expected), input[i], "Mismatch at index: ", i);
      }
    }
  }

 protected:
   std::vector<int> recvCounts_;
};

} // namespace

// Namespace for the new style algorithm benchmarks.
namespace {

template <typename T>
class NewAllreduceBenchmark : public Benchmark<T> {
  using allocation = std::vector<std::vector<T, aligned_allocator<T, kBufferAlignment>>>;//allocation 是一个别名，指向一个 std::vector 构成的数组

 public:
  NewAllreduceBenchmark(
    std::shared_ptr<::gloo::Context>& context,
    struct options& options)
      : Benchmark<T>(context, options),
        opts_(context) {}

  allocation newAllocation(int inputs, size_t elements) {//输入数据个数,输入数据元素数量，可以理解为多个矩阵中包含的多元素
    allocation out;
    out.reserve(inputs);//指定数量的空间以容纳输入数据
    for (size_t i = 0; i < inputs; i++) {
      out.emplace_back(elements);//emplace_back() 是 std::vector 类的一个成员函数，用于在容器的末尾插入长度为 elements 的、元素类型为 T de 新元素
    }
    return out;
  }

  void initialize(size_t elements) override {
    inputAllocation_ = newAllocation(this->options_.inputs, elements);//初始化输入和输出分配内存的容器
    outputAllocation_ = newAllocation(this->options_.inputs, elements);//申请空间
//yedcode 
                //  std::cout << this->options_.inputs<< "this->options_.inputs"; 
                //  std::cout << elements<< "elements"; 
//yedcode 
    // Stride between successive values in any input.
    const auto stride = this->context_->size * this->options_.inputs;
    for (size_t i = 0; i < this->options_.inputs; i++) {
      // Different for every input at every node. This means all
      // values across all inputs and all nodes are different and we
      // can accurately detect correctness errors.
      const auto value = (this->context_->rank * this->options_.inputs) + i;//value能保证不同rank的区域不一样
      for (size_t j = 0; j < elements; j++) {
        inputAllocation_[i][j] = (j * stride) + value;//有点像定位，每个节点的每个容器的位置都分配了个key
      }
    }

    // Generate vectors with pointers to populate the options struct.
    std::vector<T*> inputPointers;
    std::vector<T*> outputPointers;
    for (size_t i = 0; i < this->options_.inputs; i++) {
      inputPointers.push_back(inputAllocation_[i].data());//把整列key压入
      outputPointers.push_back(outputAllocation_[i].data());
    }

    // Configure AllreduceOptions struct
    opts_.setInputs(inputPointers, elements);
    opts_.setOutputs(outputPointers, elements);
    opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
    void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
    opts_.setReduceFunction(fn);
  }

  void run() override {
    allreduce(opts_);
  }

 private:
  AllreduceOptions opts_;

  allocation inputAllocation_;
  allocation outputAllocation_;
};

/**
 * SHArP Allreduce
*/
template <typename T>
class SharpAllreduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    SharpAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
      //opts_.elements = elements;
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements);
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        outputPointers[i] = 0;
      }
      
      for(int i=0; i<1000; i++){
        gloo::sharp_allreduce(opts_);
      }
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::sharp_allreduce(opts_);
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      for (int i = 0; i < opts_.elements; i++) {
        GLOO_ENFORCE_EQ(
          T(i + expected), outputPointers[i],
          "Mismatch at index: ", i);
      }
    }

  protected:
    SharpAllreduceOptions opts_;
    float* inputPointers;
    float* outputPointers;


};

//yed pipeallreduce
template <typename T>
class PipeAllreduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    PipeAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context,context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
      opts_.elements = elements;
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements);
      opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts_.setReduceFunction(fn);
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        outputPointers[i] = 0;
      }
      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
        gloo::pipe_allreduce(opts_);
      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::pipe_allreduce(opts_);
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      for (int i = 0; i < opts_.elements; i++) {
        GLOO_ENFORCE_EQ(
          T(i + expected), outputPointers[i],
          "Mismatch at index: ", i);
      }
    }

  protected:
    PipeAllreduceOptions opts_;
    float* inputPointers;
    float* outputPointers;

};

//yed ntallreduce
template <typename T>
class ntAllreduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    ntAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context) ,
          opts2_(context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
        size_t elements1;
        size_t elements2;         
       if(elements<32769){
    int cout_ele=2;
    int cout_mode;
    cout_mode=elements%cout_ele;
    if(cout_mode==0){
      elements2=elements/cout_ele;//2是allreduce
      elements1=elements-elements2;
    }
    else{
      elements2=(elements-cout_mode)/cout_ele;
      elements1=elements-elements2;
    }   
}
    else if(32769<elements<4194305 ){
    int cout_ele=3;
    int cout_mode;
    cout_mode=elements%cout_ele;
    if(cout_mode==0){
      elements2=elements/cout_ele;//2是allreduce
      elements1=elements-elements2;
    }
    else{
      elements2=(elements-cout_mode)/cout_ele;
      elements1=elements-elements2;
    }   
}
else{
      int cout_ele=5;
    int cout_mode;
    cout_mode=elements%cout_ele;
    if(cout_mode==0){
      elements2=2*elements/cout_ele;//2是allreduce
      elements1=elements-elements2;
    }
    else{
      elements2=2*(elements-cout_mode)/cout_ele;
      elements1=elements-elements2;
    }   
}
     
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      float* inputPointers1 =  inputPointers;
      float* outputPointers1 = outputPointers;      
      float* inputPointers2 =  inputPointers+elements1;
      float* outputPointers2 = outputPointers+elements1;

      opts_.setInput(inputPointers1, elements1);
      opts_.setOutput(outputPointers1, elements1);
      opts2_.setInput(inputPointers2, elements2);
      opts2_.setOutput(outputPointers2, elements2);

      opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts_.setReduceFunction(fn);
  
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        outputPointers[i] = 0;
      }
      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
    gloo::sharp_allreduce(opts2_);
    gloo::allreduce(opts_);

      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
  std::thread thread1([this]() {
    gloo::sharp_allreduce(opts2_);
  });

  std::thread thread2([this]() {
    gloo::allreduce(opts_);
  });

  thread1.join();
  thread2.join();
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }
    // ~ntAllreduceBenchmark() {
    //   free(inputPointers);
    //   free(outputPointers);
    // }
  protected:
    AllreduceOptions opts_;
    SharpAllreduceOptions opts2_;
    float* inputPointers;
    float* outputPointers;

};

template <typename T>
class aAllreduceBenchmark : public newBenchmark<T> {
  using newBenchmark<T>::newBenchmark;
  public:
    aAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,std::shared_ptr<::gloo::Context>& context2,
      struct options& options)
        : newBenchmark<T>(context,context2, options),
          opts_(context,context2,(this->options_.contextIbname).c_str()) {}
          

    void initialize(size_t elements) override {
      // Create input/output buffers
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      // outputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      if(elements!=0){
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements);
      }
      opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts_.setReduceFunction(fn);

      //  printf("\n allreduce begin\n");
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        // outputPointers[i] = 0;
      }
      for (int i = 0; i < elements; i++) {
        outputPointers[i] = 0;
      }

      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
      gloo::apipe_allreduce(opts_);  
      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::apipe_allreduce(opts_);
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }
    // ~ntAllreduceBenchmark() {
    //   free(inputPointers);
    //   free(outputPointers);
    // }
  protected:
    APipeAllreduceOptions opts_;
    float* inputPointers;
    float* outputPointers;
};
template <typename T>
class sAllreduceBenchmark : public newBenchmark<T> {
  using newBenchmark<T>::newBenchmark;
  public:
    sAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,std::shared_ptr<::gloo::Context>& context2,
      struct options& options)
        : newBenchmark<T>(context,context2, options),
          opts_(context,context2,(this->options_.contextIbname).c_str()) {}
          

    void initialize(size_t elements) override {
      // Create input/output buffers
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      // outputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      if(elements!=0){
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements);
      }
      opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts_.setReduceFunction(fn);

      //  printf("\n allreduce begin\n");
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        // outputPointers[i] = 0;
      }
      for (int i = 0; i < elements; i++) {
        outputPointers[i] = 0;
      }

      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
      gloo::spipe_allreduce(opts_);  
      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::spipe_allreduce(opts_);
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }
    // ~ntAllreduceBenchmark() {
    //   free(inputPointers);
    //   free(outputPointers);
    // }
  protected:
    SPipeAllreduceOptions opts_;
    float* inputPointers;
    float* outputPointers;
};


template <typename T>
class btAllreduceBenchmark : public newBenchmark<T> {
  using newBenchmark<T>::newBenchmark;
  public:
    btAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,std::shared_ptr<::gloo::Context>& context2,
      struct options& options)
        : newBenchmark<T>(context,context2, options),
          opts_(context,context2,(this->options_.contextIbname).c_str()) {}
          

    void initialize(size_t elements) override {
      // Create input/output buffers
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      // outputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      if(elements!=0){
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements);
      }
      opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts_.setReduceFunction(fn);

      //  printf("\n allreduce begin\n");
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        // outputPointers[i] = 0;
      }
      for (int i = 0; i < elements; i++) {
        outputPointers[i] = 0;
      }

      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
      gloo::pipe_allreduce(opts_);  
      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::pipe_allreduce(opts_);
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }
    // ~ntAllreduceBenchmark() {
    //   free(inputPointers);
    //   free(outputPointers);
    // }
  protected:
    PipeAllreduceOptions opts_;
    float* inputPointers;
    float* outputPointers;
};


template <typename T>
class stAllreduceBenchmark : public newBenchmark<T> {
  using newBenchmark<T>::newBenchmark;
  public:
    stAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,std::shared_ptr<::gloo::Context>& context2,
      struct options& options)
        : newBenchmark<T>(context,context2, options),
          opts2_(context2),
          opts1_(context, (this->options_.contextIbname).c_str()) {}
          

    void initialize(size_t elements) override {
      // Create input/output buffers
        size_t elements1;
        size_t elements2;     
        int cout_mode;   
        int cout_ele=100;
        int w_2;    
       if(getenv("SHARP_GLEX") != NULL && this->context_->size==4){
        cout_ele=100;
        if(elements<14935){
                cout_ele=1;
                w_2=0;
            }
        else if(14934<elements && elements<131073){
                cout_ele=1;
                w_2=1;
        }          
        else if(131072<elements && elements<262145){
                w_2=70;
        }                   
        else if(262144<elements && elements<524289){
                w_2=70;
        }          
        else if(524288<elements && elements<1048577){
                w_2=68;                
        }           
        else if(1048576<elements && elements<2097153){
                w_2=68;
        }            
        else if(2097152<elements && elements<4194305){
                w_2=69;             
        }    
        else if(4194304<elements && elements<8388609){    
                w_2=66;       
        }                       
        else if(8388608<elements && elements<16777217){
                w_2=61;       
        }          
        else if(16777216<elements && elements<33554433){ 
                w_2=59;       
        }     
        else if(33554432<elements && elements<67108865){
                w_2=58;       
        }         
        else{
                cout_ele=2;
                w_2=1;
        }
    }
    else if(getenv("SHARP_GLEX") != NULL && this->context_->size==2){
        int cout_mode;   
        int cout_ele=100;
        int w_2;
        if(elements<4097){
                cout_ele=1;                
                w_2=0;
            }
        else if(4096<elements && elements<524289){
                cout_ele=1;                
                w_2=1;
        }       
        else if(524288<elements && elements<1048577){
                w_2=76;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=75;
        }          
        else if(2097152<elements && elements<4194305){
                w_2=75;                
        }             
        else if(4194304<elements && elements<8388609){
                w_2=75;                
        }       
        else if(8388608<elements && elements<16777217){
                w_2=70;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=66;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=66;                
        }   
        else{
                w_2=50;
        }

    }
    else if(getenv("SHARP_ALLREDUCE") != NULL && this->context_->size==2){
        int cout_mode;   
        int cout_ele=100;
        int w_2;
        if(elements<65537){
                w_2=0;
            }
        else if(65536<elements && elements<131073){
                w_2=28;
        } 
        else if(131072<elements && elements<262145){
                w_2=35;
        }        
        else if(262144<elements && elements<524289){
                w_2=48;

        }     
        else if(524288<elements && elements<1048577){
                w_2=49;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=49;
        }          
        else if(2097152<elements && elements<4194305){
                w_2=49;                
        }             
        else if(4194304<elements && elements<8388609){
                w_2=46;                
        }       
        else if(8388608<elements && elements<16777217){
                w_2=48;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=45;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=46;                
        }   
        else{
                w_2=1;
        }
    }
    else if(getenv("SHARP_GLEX") != NULL && this->context_->size==3){

        if(elements<8193){
                cout_ele=1;
                w_2=0;
            }
        else if(8192<elements && elements<65537){
                cout_ele=1;
                w_2=1;
        }       
        else if(65536<elements && elements<131073){
                cout_ele=13;
                w_2=10;
        }          
        else if(131072<elements && elements<262145){
                cout_ele=19;
                w_2=14;
        }                   
        else if(262144<elements && elements<524289){
                cout_ele=69;
                w_2=49;
        }          
        else if(524288<elements && elements<1048577){
                cout_ele=64;
                w_2=45;                
        }           
        else if(1048576<elements && elements<2097153){
                cout_ele=64;
                w_2=43;
        }            
        else if(2097152<elements && elements<4194305){
                cout_ele=47;
                w_2=30;             
        }    
        else if(4194304<elements && elements<8388609){    
                cout_ele=57;
                w_2=35;       
        }                       
        else if(8388608<elements && elements<16777217){ 
                cout_ele=71;
                w_2=42;       
        }          
        else if(16777216<elements && elements<33554433){
                cout_ele=7;
                w_2=4;       
        }     
        else if(33554432<elements && elements<67108865){
                cout_ele=25;
                w_2=14;       
        }      
        else{
                cout_ele=1;
                w_2=1;
        } 
    }
    else if(getenv("SHARP_ALLREDUCE") != NULL &&this->context_->size==3){
        if(elements<131073){
                cout_ele=1;
                w_2=0;
            }
        else if(131072<elements && elements<262145){
                cout_ele=47;
                w_2=9;
        }        
        else if(262144<elements && elements<524289){
                cout_ele=71;
                w_2=20;
        }     
        else if(524288<elements && elements<1048577){
                cout_ele=26;
                w_2=11;
        }  
        else if(1048576<elements && elements<2097153){
                cout_ele=73;
                w_2=30;
        }          
        // else if(2097152<elements && elements<4194305){
        //         cout_ele=52;
        //         w_2=21;                
        // }             
        // else if(4194304<elements && elements<8388609){
        //         cout_ele=52;
        //         w_2=21;                
        // }       
        // else if(8388608<elements && elements<16777217){
        //         cout_ele=5;
        //         w_2=2;                
        // }          
        // else if(16777216<elements && elements<33554433){
        //         cout_ele=9;
        //         w_2=4;                
        // }     
        else if(2097152<elements && elements<67108865){
                cout_ele=5;
                w_2=2;                
        }         

        else{
                cout_ele=2;
                w_2=1;
        }
    }

            cout_mode = elements % cout_ele;
      if (cout_mode == 0) {
              elements2 = w_2 * elements / cout_ele;
              elements1 = elements - elements2;
      }
      else {
              elements2 = w_2 * (elements - cout_mode) / cout_ele;
              elements1 = elements - elements2;
    }  
opts1_.elements=elements1;
opts2_.getImpl().elements=elements2;


      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements1));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      inputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      // outputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      float* outputPointers2 = outputPointers+elements1;
      if(elements2!=0){
      opts2_.setInput(inputPointers2, elements2);
      opts2_.setOutput(outputPointers2, elements2);
      opts2_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;
      opts2_.setReduceFunction(fn);
      }

      if(elements1!=0){
      opts1_.setInput(inputPointers, elements1);
      opts1_.setOutput(outputPointers, elements1);  }

      for (int i = 0; i < elements1; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        // outputPointers[i] = 0;
      }
      for (int i = 0; i < elements2; i++) {
        inputPointers2[i] = (i+elements1) * (this->context_->rank + 1.0);
        // outputPointers2[i] = 0;
      }
      for (int i = 0; i < elements; i++) {
        outputPointers[i] = 0;
      }

      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
          std::future<void> fut1;
          std::future<void> fut2;          
          if (elements1 != 0 && elements2 != 0) {
              std::future<void> fut1 = std::async(std::launch::async, [=] {
                  gloo::allreduce(opts2_);
              });
              std::future<void> fut2 = std::async(std::launch::async, [=] {
                  gloo::sharp_allreduce(opts1_);
              });
              fut1.wait();
              fut2.wait();
          }
          else if (elements2 != 0&&elements1 == 0){
                  gloo::allreduce(opts2_);
          }
          else{
                  gloo::sharp_allreduce(opts1_);            
          }      

      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
    if (opts2_.getImpl().elements != 0&&opts1_.elements != 0) {
        std::thread thread1([this]() {
          gloo::sharp_allreduce(opts1_);
        });
        std::thread thread2([this]() {
          gloo::allreduce(opts2_);
        });
        thread1.join();
        thread2.join();     
          }     else if (opts2_.getImpl().elements != 0&&opts1_.elements == 0){
                        gloo::allreduce(opts2_);
                }
                else{
                        gloo::sharp_allreduce(opts1_);            
                }  
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }
    // ~ntAllreduceBenchmark() {
    //   free(inputPointers);
    //   free(outputPointers);
    // }
  protected:
    AllreduceOptions opts2_;
    SharpAllreduceOptions opts1_;
    float* inputPointers;
    float* outputPointers;
    float* inputPointers2;
    float* outputPointers2;
};

template <typename T>
class agAllreduceBenchmark : public newBenchmark<T> {
  using newBenchmark<T>::newBenchmark;
  public:
    agAllreduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,std::shared_ptr<::gloo::Context>& context2,
      struct options& options)
        : newBenchmark<T>(context,context2, options),
          opts_(context),
          opts2_(context2) {}
          

    void initialize(size_t elements) override {
      // Create input/output buffers
        size_t elements1;
        size_t elements2;         
       if(this->context_->size==4){
            if(elements<32767){
                elements2=elements/2;//2是allreduce
                elements1=elements/2;
            }
            else if(elements==32768){
                elements2=elements/2;//2是allreduce
                elements1=elements/2;
            }            
            else if(32768<elements<4194305 ){
                int cout_ele=3;
                int cout_mode;
                cout_mode=elements%cout_ele;
                if(cout_mode==0){
                    elements2=elements/cout_ele;//2是allreduce
                    elements1=elements-elements2;
                }
                else{
                    elements2=(elements-cout_mode)/cout_ele;
                    elements1=elements-elements2;
                    }   
            }
            else if(4194304<elements<67108864 ){
                    int cout_ele=5;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements2=2*elements/cout_ele;//2是allreduce
                    elements1=elements-elements2;
                  }
            else{
              elements2=2*(elements-cout_mode)/cout_ele;
              elements1=elements-elements2;
                 }   
            }
            else{
                    int cout_ele=17;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements2=7*elements/cout_ele;//2是allreduce
                    elements1=elements-elements2;
                  }
            else{
              elements2=7*(elements-cout_mode)/cout_ele;
              elements1=elements-elements2;
                 }   
            }
    }
    else if(this->context_->size==2){
            if(elements<32769){
                elements2=elements;//2是allreduce
                elements1=0;
            }
            else if(32769<elements<4194305 ){
                int cout_ele=3;
                int cout_mode;
                cout_mode=elements%cout_ele;
                if(cout_mode==0){
                    elements2=elements/cout_ele;//2是allreduce
                    elements1=elements-elements2;
                }
                else{
                    elements2=(elements-cout_mode)/cout_ele;
                    elements1=elements-elements2;
                    }   
            }
            else{
                    int cout_ele=5;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements2=2*elements/cout_ele;//2是allreduce
                    elements1=elements-elements2;
                  }
            else{
              elements2=2*(elements-cout_mode)/cout_ele;
              elements1=elements-elements2;
                 }   
            }
    }


opts2_.getImpl().elements=elements2;
opts_.getImpl().elements=elements1;
    // printf("\n allreduce begin1\n");

      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements1));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      inputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      // outputPointers2 = reinterpret_cast<float*>(malloc(sizeof(float) * elements2));
      float* outputPointers2 = outputPointers+elements1;
      void (*fn)(void*, const void*, const void*, long unsigned int) = &sum<T>;      
      if(elements1!=0){
      opts_.setInput(inputPointers, elements1);
      opts_.setOutput(outputPointers, elements1);
      opts_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      opts_.setReduceFunction(fn);
          // printf("\n allreduce begin2\n");
      }

      if(elements2!=0){
      opts2_.setInput(inputPointers2, elements2);
      opts2_.setOutput(outputPointers2, elements2); 
      opts2_.setAlgorithm(AllreduceOptions::Algorithm::RING);
      opts2_.setReduceFunction(fn);     
          // printf("\n allreduce begin2\n");
      
       }

      for (int i = 0; i < elements1; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        // outputPointers[i] = 0;
      }
      for (int i = 0; i < elements2; i++) {
        inputPointers2[i] = (i+elements1) * (this->context_->rank + 1.0);
        // outputPointers2[i] = 0;
      }
      for (int i = 0; i < elements; i++) {
        outputPointers[i] = 0;
      }

      // printf("\n allreduce begin\n");
      for(int i=0; i<1000; i++){
          std::future<void> fut1;
          std::future<void> fut2;          
          if (elements1 != 0 && elements2 != 0) {
              std::future<void> fut1 = std::async(std::launch::async, [=] {
                  gloo::allreduce(opts_);
              });
              std::future<void> fut2 = std::async(std::launch::async, [=] {
                  gloo::allreduce(opts2_);
              });
              fut1.wait();
                  // printf("\n allreduce begin3\n");
              fut2.wait();
                  // printf("\n allreduce begin4\n");
          }
          else if (elements1 != 0&&elements2 == 0){
                  gloo::allreduce(opts_);
          }
          else{
                  gloo::allreduce(opts2_);            
          }      

      }
      // printf("allreduce end\n");
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
    if (opts2_.getImpl().elements!=0 && opts_.getImpl().elements!=0) {
        std::thread thread1([this]() {
          gloo::allreduce(opts2_);
        });
        std::thread thread2([this]() {
          gloo::allreduce(opts_);
        });
        thread1.join();
        thread2.join();     
          }     else if (opts_.getImpl().elements != 0&&opts2_.getImpl().elements == 0){
                        gloo::allreduce(opts_);
                }
                else{
                        gloo::allreduce(opts2_);            
                }  
    }

    void verify() override {
      const int rank = this->context_->rank;
      const int size = this->context_->size;
      auto expected = 0;
      for(int i = 0; i < size; i++){
        expected += (i + 1);
      }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }
    // ~ntAllreduceBenchmark() {
    //   free(inputPointers);
    //   free(outputPointers);
    // }
  protected:
    AllreduceOptions opts_;
    AllreduceOptions opts2_;
    float* inputPointers;
    float* outputPointers;
    float* inputPointers2;
    float* outputPointers2;
};



/**
 * SHArP Reduce
*/
template <typename T>
class SharpReduceBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    SharpReduceBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      opts_.setInput(inputPointers, elements);

      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      opts_.setOutput(outputPointers, elements);
      
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = (float)(i * (this->context_->rank + 1.0));
      }

      for(int i=0; i<100; i++){
        gloo::sharp_reduce(opts_);
      }
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::sharp_reduce(opts_);
    }

    void verify() override {
      // const int rank = this->context_->rank;
      // const int size = this->context_->size;
      // auto expected = 0;
      // for(int i = 0; i < size; i++){
      //   expected += (i + 1);
      // }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }

  protected:
    SharpReduceOptions opts_;
    float* inputPointers;
    float* outputPointers;
};

/**
 * SHArP Broadcast
*/
template <typename T>
class SharpBroadcastBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    SharpBroadcastBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
      bufPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      opts_.setBufer(bufPointers, elements);
      if(this->context_->rank == 0){
        for (int i = 0; i < elements; i++) {
          bufPointers[i] = (float)(i * (this->context_->rank + 1.0));
        }
      }


      for(int i=0; i<100; i++){
        gloo::sharp_broadcast(opts_);
      }
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::sharp_broadcast(opts_);
    }

    void verify() override {
      // const int rank = this->context_->rank;
      // const int size = this->context_->size;
      // auto expected = 0;
      // for(int i = 0; i < size; i++){
      //   expected += (i + 1);
      // }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }

  protected:
    SharpBroadcastOptions opts_;
    float* bufPointers;
};

/**
 * SHArP Allgather
*/
template <typename T>
class SharpAllgatherBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    SharpAllgatherBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      // Create input/output buffers
      //opts_.elements = elements;
      inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements));
      outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * elements * (this->context_->size)));
      for (int i = 0; i < elements; i++) {
        inputPointers[i] = i * (this->context_->rank + 1.0);
        outputPointers[i] = 0;
      }
      opts_.setInput(inputPointers, elements);
      opts_.setOutput(outputPointers, elements * (this->context_->size));

      
      for(int i=0; i<1000; i++){
        gloo::sharp_allgather(opts_);
      }
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::sharp_allgather(opts_);
    }

    void verify() override {
      // const int rank = this->context_->rank;
      // const int size = this->context_->size;
      // auto expected = 0;
      // for(int i = 0; i < size; i++){
      //   expected += (i + 1);
      // }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }

  protected:
    SharpAllgatherOptions opts_;
    float* inputPointers;
    float* outputPointers;


};

/**
 * SHArP Barrier
*/
template <typename T>
class SharpBarrierBenchmark : public Benchmark<T> {
  using Benchmark<T>::Benchmark;
  public:
    SharpBarrierBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      struct options& options)
        : Benchmark<T>(context, options),
          opts_(context,(this->options_.contextIbname).c_str()) {}

    void initialize(size_t elements) override {
      for(int i=0; i<1000; i++){
        gloo::sharp_barrier(opts_);
      }
    }

    // Default run function calls Algorithm::run
    // Need to override this function for collectives that
    // do not inherit from the Algorithm class
    void run() override {
      // Run the collective on the previously created options      
      gloo::sharp_barrier(opts_);
    }

    void verify() override {
      // const int rank = this->context_->rank;
      // const int size = this->context_->size;
      // auto expected = 0;
      // for(int i = 0; i < size; i++){
      //   expected += (i + 1);
      // }

      // for (int i = 0; i < opts_.elements; i++) {
      //   GLOO_ENFORCE_EQ(
      //     T(i + expected), outputPointers[i],
      //     "Mismatch at index: ", i);
      // }
    }

  protected:
    SharpBarrierOptions opts_;
    float* inputPointers;
    float* outputPointers;


};

}

#define RUN_BENCHMARK(T)                                                   \
  Runner::BenchmarkFn<T> fn;                                               \
  if (x.benchmark == "allgather_ring") {                                   \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<AllgatherBenchmark<T>>(context, x);         \
    };                                                                     \
  } else if (x.benchmark == "allreduce_ring") {                            \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<AllreduceBenchmark<AllreduceRing<T>, T>>(   \
          context, x);                                                     \
    };                                                                     \
  } else if (x.benchmark == "allreduce_ring_chunked") {                    \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceRingChunked<T>, T>>(context, x);     \
    };                                                                     \
  } else if (x.benchmark == "allreduce_halving_doubling") {                \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceHalvingDoubling<T>, T>>(context, x); \
    };                                                                     \
  } else if (x.benchmark == "allreduce_bcube") {                           \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<                                            \
          AllreduceBenchmark<AllreduceBcube<T>, T>>(context, x);           \
    };                                                                     \
  } else if (x.benchmark == "barrier_all_to_all") {                        \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BarrierAllToAllBenchmark<T>>(context, x);   \
    };                                                                     \
  } else if (x.benchmark == "barrier_all_to_one") {                        \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BarrierAllToOneBenchmark<T>>(context, x);   \
    };                                                                     \
  } else if (x.benchmark == "broadcast_one_to_all") {                      \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<BroadcastOneToAllBenchmark<T>>(context, x); \
    };                                                                     \
  } else if (x.benchmark == "reduce") {                                    \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<ReduceBenchmark<T>>(context, x);            \
    };                                                                     \
  } else if (x.benchmark == "pairwise_exchange") {                         \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<PairwiseExchangeBenchmark<T>>(context, x);  \
    };                                                                     \
  } else if (x.benchmark == "reduce_scatter") {                            \
    fn = [&](std::shared_ptr<Context>& context) {                          \
      return gloo::make_unique<ReduceScatterBenchmark<T>>(context, x);  \
    };                                                                     \
  }                                                                        \
  if (!fn) {                                                               \
    GLOO_ENFORCE(false, "Invalid algorithm: ", x.benchmark);               \
  }                                                                        \
  Runner r(x);                                                             \
  r.run(fn);

template <typename T>
void runNewBenchmark(options& options) {
  Runner::BenchmarkFn<T> fn;

  const auto name = options.benchmark.substr(4);
  if (name == "allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<NewAllreduceBenchmark<T>>(context, options);
    };
  } else if(name == "allreduce_sharp") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<SharpAllreduceBenchmark<T>>(context, options);
    };
  } else if(name == "allreduce_pipe") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<PipeAllreduceBenchmark<T>>(context, options);
    };    
  } else if(name == "allreduce_nt") {
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<ntAllreduceBenchmark<T>>(context, options);
    };        
  // } else if(name == "allreduce_bt") {
  //   fn = [&](std::shared_ptr<Context>& context) {
  //     return gloo::make_unique<btAllreduceBenchmark<T>>(context, options);
  //   };            
  } else if(name == "reduce_sharp"){
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<SharpReduceBenchmark<T>>(context, options);
    };
  } else if(name == "bcast_sharp"){
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<SharpBroadcastBenchmark<T>>(context, options);
    };
  } else if(name == "allgather_sharp"){
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<SharpAllgatherBenchmark<T>>(context, options);
    };
  } else if(name == "barrier_sharp"){
    fn = [&](std::shared_ptr<Context>& context) {
      return gloo::make_unique<SharpBarrierBenchmark<T>>(context, options);
    };
  } else {
    GLOO_ENFORCE(false, "Invalid benchmark name: ", options.benchmark);
  }

  Runner runner(options);
  runner.run(fn);
}
#if GLOO_HAVE_TRANSPORT_GLEX_RDMA_S
template <typename T>
void runbewBenchmark(options& options) {
  Runner::BenchmarkFn2<T> fn;

  const auto name = options.benchmark.substr(4);
  if (name == "allreduce_bt") {
    fn = [&](std::shared_ptr<Context>& context,std::shared_ptr<Context>& context2) {
      return gloo::make_unique<btAllreduceBenchmark<T>>(context,context2, options);
    };
  }  else if (name == "allreduce_st") {
    fn = [&](std::shared_ptr<Context>& context,std::shared_ptr<Context>& context2) {
      return gloo::make_unique<stAllreduceBenchmark<T>>(context,context2, options);
    };
  }  else if (name == "allreduce_s") {
    fn = [&](std::shared_ptr<Context>& context,std::shared_ptr<Context>& context2) {
      return gloo::make_unique<sAllreduceBenchmark<T>>(context,context2, options);
    };    
  }  else if (name == "allreduce_a") {
    fn = [&](std::shared_ptr<Context>& context,std::shared_ptr<Context>& context2) {
      return gloo::make_unique<aAllreduceBenchmark<T>>(context,context2, options);
    };        
  }  else if (name == "allreduce_ag") {
    fn = [&](std::shared_ptr<Context>& context,std::shared_ptr<Context>& context2) {
      return gloo::make_unique<agAllreduceBenchmark<T>>(context,context2, options);
    };    
  }     else {
    GLOO_ENFORCE(false, "Invalid benchmark name: ", options.benchmark);
  }

  Runner runner(options);
  runner.run2(fn);
}
#endif



int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);
  std::shared_ptr<gloo::rendezvous::Context> context2;

  // Run new style benchmarks if the benchmark name starts with "new_".
  // Eventually we'd like to deprecate all the old style ones...
  if (x.benchmark.substr(0, 4) == "new_") {
    runNewBenchmark<float>(x);
    return 0;
  }
#if GLOO_HAVE_TRANSPORT_GLEX_RDMA_S
  if (x.benchmark.substr(0, 4) == "bew_") {
    runbewBenchmark<float>(x);
    return 0;
  }
#endif

  if (x.benchmark == "pairwise_exchange") {
    RUN_BENCHMARK(char);
  } else if (x.halfPrecision) {
    RUN_BENCHMARK(float16);
  } else {
    RUN_BENCHMARK(float);
  }
  return 0;
}
