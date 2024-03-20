/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include <iostream>
#include <memory>
#include <array>
#include <typeinfo>
#include <string>

#include "gloo/sharp_allreduce.h"
#include "gloo/pipeallreduce.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/glex/device.h"
#include "gloo/allreduce.h"
#include "gloo/transport/glex_rdma_t/device.h"

#include "gloo/rendezvous/redis_store.h"

#ifndef GLOO_HAVE_TRANSPORT_IBVERBS
#define GLOO_HAVE_TRANSPORT_IBVERBS
#endif
// Usage:
//
// Open two terminals. Run the same program in both terminals, using
// a different RANK in each. For example:
//
// A: PREFIX=test1 SIZE=2 RANK=0 example_allreduce
// B: PREFIX=test1 SIZE=2 RANK=1 example_allreduce
//
// Expected output:
//
//   data[0] = 18
//   data[1] = 18
//   data[2] = 18
//   data[3] = 18
//

void mysum(void* c_, const void* a_, const void* b_, int n) {
  printf("n=%d\r\n", n);
  float* c = static_cast<float*>(c_);
  const float* a = static_cast<const float*>(a_);
  const float* b = static_cast<const float*>(b_);
  for (auto i = 0; i < n; i++) {
    printf("a[%d]=%f\r\n", i, a[i]);
    printf("b[%d]=%f\r\n", i, b[i]);
    c[i] = a[i] + b[i];
    printf("c[%d]=%f\r\n", i, c[i]);
  }
}

int main(void) {
  if (getenv("PREFIX") == nullptr ||
      getenv("SIZE") == nullptr ||
      getenv("RANK") == nullptr ||
      getenv("HOST") == nullptr ||
      getenv("IB_NAME") == nullptr) {
    std::cerr
      << "Please set environment variables PREFIX, SIZE, RANK and IB_NAME."
      << std::endl;
    return 1;
  }

  // The following statement creates a TCP "device" for Gloo to use.
  // See "gloo/transport/device.h" for more information. For the
  // purposes of this example, it is sufficient to see the device as
  // a factory for every communication pair.
  
  // The argument to gloo::transport::tcp::CreateDevice is used to
  // find the network interface to bind connection to. The attr struct
  // can be populated to specify exactly which interface should be
  // used, as shown below. This is useful if you have identical
  // multi-homed machines that all share the same network interface
  // name, for example.
  //

  // gloo::transport::tcp::attr attr;
  // attr.iface = "eth0";
  // // attr.iface = "ib0";
  // // attr.iface = "Wi-Fi";

  // // attr.ai_family = AF_INET; // Force IPv4
  // // attr.ai_family = AF_INET6; // Force IPv6
  // attr.ai_family = AF_UNSPEC;  // Use either (default)


  // A string is implicitly converted to an "attr" struct with its
  // hostname field populated. This will try to resolve the interface
  // to use by resolving the hostname or IP address, and finding the
  // corresponding network interface.
  
  // Hostname "localhost" should resolve to 127.0.0.1, so using this
  // implies that all connections will be local. This can be useful
  // for single machine operation.
  
  //   auto dev = gloo::transport::tcp::CreateDevice("localhost");
  //
// //yed
//  auto dev = gloo::transport::tcp::CreateDevice(attr);
// //
  auto dev = gloo::transport::glex_rdma_t::CreateDevice();

  fprintf(stdout, "GLEX_RDMA_T Device created.\n");
  // auto dev = gloo::transport::glex_rdma_t::CreateDevice();

  // fprintf(stdout, "GLEX_RDMA_T Device created.\n");
    // auto dev = gloo::transport::glex::CreateDevice();
  // Now that we have a device, we can connect all participating
  // processes. We call this process "rendezvous". It can be performed
  // using a shared filesystem, a Redis instance, or something else by
  // extending it yourself.
  
  // See "gloo/rendezvous/store.h" for the functionality you need to
  // implement to create your own store for performing rendezvous.
  
  // Below, we instantiate rendezvous using the filesystem, given that
  // this example uses multiple processes on a single machine.
  std::string redisHost = getenv("HOST");
  gloo::rendezvous::RedisStore redisStore(redisHost);

  
  auto fileStore = gloo::rendezvous::FileStore("/home/mpi_share/env/gloo_etc/tmp");

  // To be able to reuse the same store over and over again and not have
  // interference between runs, we scope it to a unique prefix with the
  // PrefixStore. This wraps another store and prefixes every key before
  // forwarding the call to the underlying store.
  std::string prefix = getenv("PREFIX");
  auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);

  // Using this store, we can now create a Gloo context. The context
  // holds a reference to every communication pair involving this
  // process. It is used by every collective algorithm to find the
  // current process's rank in the collective, the collective size,
  // and setup of send/receive buffer pairs.
  const int rank = atoi(getenv("RANK"));
  const int size = atoi(getenv("SIZE"));
  const std::string ibname = getenv("IB_NAME");
  
  auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
  context->connectFullMesh(prefixStore, dev);

  printf("rank:%d size:%d connected full mesh\n", rank, size);

  // All connections are now established. We can now initialize some
  // test data, instantiate the collective algorithm, and run it.
 //yed correct1
// /*
  float *inputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * 10));
  float *outputPointers = reinterpret_cast<float*>(malloc(sizeof(float) * 10));

  // std::array<float, 100> inputPointers;
  // std::array<float, 100> outputPointers;



  gloo::PipeAllreduceOptions opts_(context,ibname.c_str());
  opts_.setInput(inputPointers, 10);
  opts_.setOutput(outputPointers, 10);
  opts_.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
  void (*fn)(void*, const void*, const void*, int) = &mysum;
  opts_.setReduceFunction(fn);
  for (int i = 0; i < 10; i++) {//压入数据
    inputPointers[i] = i * (rank + 1.0);
    outputPointers[i] = 0;
  }

  std::cout << "Input: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << "data[" << i << "] = " << inputPointers[i] << std::endl;
  }

  gloo::pipe_allreduce(opts_);

// */
//yed correct2
/*
  size_t elements = 4;
  std::vector<int*> inputPointers;
  std::vector<int*> outputPointers;
  for (size_t i = 0; i < elements; i++) {
    int *value = reinterpret_cast<int*>(malloc(sizeof(int)));
    *value = i * (rank + 1);
    inputPointers.push_back(value);
    int *value1 = reinterpret_cast<int*>(malloc(sizeof(int)));
    *value1 = 0;
    outputPointers.push_back(value1);
  }

  // Configure AllreduceOptions struct
  gloo::PipeAllreduceOptions opts_(context,ibname.c_str());
  opts_.setInputs(inputPointers, 1);
  opts_.setOutputs(outputPointers, 1);
  opts_.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
  void (*fn)(void*, const void*, const void*, int) = &mysum;
  opts_.setReduceFunction(fn);
  // gloo::sharp_allreduce_init(opts_);// yed ?
  gloo::pipe_allreduce(opts_);
  // gloo::sharp_allreduce_destroy();//yed ?
  */

  // Print the result.
  std::cout << "Output: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << "data[" << i << "] = " << outputPointers[i] << std::endl;
  }

  return 0;
}
