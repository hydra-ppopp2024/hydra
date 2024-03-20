
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

// std::chrono::microseconds totalDurationSharp;
// std::chrono::microseconds totalDurationAllReduce;
// int sharpCount = 0;
// int allReduceCount = 0;
// int printRatio(int ratio) {
//   // 在这里可以使用传递进来的 ratio 值，进行输出或其他操作
//   std::cout << "ratio: " << ratio << "ms" << std::endl;
//   return ratio;
// }

namespace gloo {
void pipe_allreduce(PipeAllreduceOptions& opts) {
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
        // if (opts.opts1.elements!=0 && opts.opts2.getImpl().elements!=0) {
        //     std::thread sharpThread([&opts]() {
        //         gloo::sharp_allreduce(opts.opts1);
        //     });
        //     std::thread allreduceThread([&opts]() {
        //         gloo::allreduce(opts.opts2);
        //     });
        //     allreduceThread.join();  
        //     sharpThread.join();
        // }    
        // else if (opts.opts2.getImpl().elements != 0&&opts.opts1.elements == 0){
        //     gloo::allreduce(opts.opts2);
        // }
        // else{
        //     gloo::sharp_allreduce(opts.opts1);   
        //     }  
    }
    else{
        if (opts.opts3.getImpl().elements!=0 && opts.opts2.getImpl().elements!=0) {
            std::thread AThread([&opts]() {
                gloo::allreduce(opts.opts3);
            });
            std::thread allreduceThread([&opts]() {
                gloo::allreduce(opts.opts2);
            });
            allreduceThread.join();  
            AThread.join();
        }    
        else if (opts.opts2.getImpl().elements != 0&&opts.opts3.getImpl().elements == 0){
            gloo::allreduce(opts.opts2);
        }
        else{
            gloo::allreduce(opts.opts3);     
            }  
    }
}

// void getElements(size_t elements,int parac) {
//     static std::unordered_map<size_t, size_t> elementCount; // 用于存储不同大小的元素和其对应的数量
//     parac++;
//     if (elementCount.find(elements) != elementCount.end()) {
//         elementCount[elements]++;
//     } else {
//         elementCount[elements] = 1;
//     }
//     // std::cout << "Received " << elements << " elements. Count: " << elementCount[elements] << std::endl;
//     if(parac>=18000){
//         std::cout << "parac: " << parac<< std::endl;       
//               for (const auto& pair : elementCount) {
//                   std::cout << "parac: " << parac<<  "Element size: " << pair.first << ", Count: " << pair.second << std::endl;
//               }    
//               }
//     }
}

