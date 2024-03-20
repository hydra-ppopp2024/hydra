
#include <assert.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <thread>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"
#include "gloo/allreduce.h"
#include "gloo/pipeallreduce.h"
#include "gloo/sharp_allreduce.h"

// std::chrono::milliseconds totalDurationSharp;
// std::chrono::milliseconds totalDurationAllReduce;
// int sharpCount = 0;
// int allReduceCount = 0;
namespace gloo {
void pipe_allreduce(PipeAllreduceOptions& opts) {

    if (opts.opts1.elements!=0 && opts.opts2.getImpl().elements!=0) {
  std::thread sharpThread([&opts]() {
      gloo::sharp_allreduce(opts.opts1);
  });
  std::thread allreduceThread([&opts]() {
      gloo::allreduce(opts.opts2);
  });
   allreduceThread.join();  
  sharpThread.join();
   }    else if (opts.opts2.getImpl().elements != 0&&opts.opts1.elements == 0){
                        gloo::allreduce(opts.opts2);
                }
                else{
                        gloo::sharp_allreduce(opts.opts1);               
                }  
}

   
}