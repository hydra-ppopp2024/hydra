
#include <assert.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <thread>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"
#include "gloo/allreduce.h"
#include "gloo/pipeallreduce-s.h"
#include "gloo/sharp_allreduce.h"

// std::chrono::microseconds totalDurationSharp;
// std::chrono::microseconds totalDurationAllReduce;
// int sharpCount = 0;
// int allReduceCount = 0;
int Count3 = 0;
int Count4 = 0;

namespace gloo {
void spipe_allreduce(SPipeAllreduceOptions& opts) {
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
        if (opts.opts1.elements!=0 && opts.opts2.getImpl().elements!=0) {
            std::thread sharpThread([&opts]() {
                auto start = std::chrono::steady_clock::now();
                gloo::sharp_allreduce(opts.opts1);
                Count3++;
                // std::cout << Count3<<"count3" << std::endl;
                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                if(Count3%500==0){
                        std::cout << opts.opts2.getImpl().elements << " elements2 "<< opts.opts1.elements<< " elements1 "<< "Allreduce1 took " << duration.count() << " milliseconds." << std::endl;
                        }
            });
            std::thread allreduceThread([&opts]() {
                auto start = std::chrono::steady_clock::now();
                gloo::allreduce(opts.opts2);
                Count4++;
                auto end = std::chrono::steady_clock::now();
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                if(Count4%500==0){
                        std::cout << opts.opts2.getImpl().elements << " elements2 "<< opts.opts1.elements<< " elements1 "<< "Allreduce2 took " << duration2.count() << " milliseconds." << std::endl;      
                }          

            });
            allreduceThread.join();  
            sharpThread.join();
        }    
        else if (opts.opts2.getImpl().elements != 0&&opts.opts1.elements == 0){
            gloo::allreduce(opts.opts2);
        }
        else{
            gloo::sharp_allreduce(opts.opts1);   
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

