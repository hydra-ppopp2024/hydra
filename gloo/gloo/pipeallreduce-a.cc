
#include <assert.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <thread>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"
#include "gloo/allreduce.h"
#include "gloo/pipeallreduce-a.h"
#include "gloo/sharp_allreduce.h"

// std::chrono::microseconds totalDurationSharp;
// std::chrono::microseconds totalDurationAllReduce;
int Count = 0;
int Count2 = 0;
// int allReduceCount = 0;
// int printRatio(int ratio) {
//   // 在这里可以使用传递进来的 ratio 值，进行输出或其他操作
//   std::cout << "ratio: " << ratio << "ms" << std::endl;
//   return ratio;
// }

namespace gloo {
void apipe_allreduce(APipeAllreduceOptions& opts) {
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
    }
    else{
        if (opts.opts3.getImpl().elements!=0 && opts.opts2.getImpl().elements!=0) {
            std::thread AThread([&opts]() {
                auto start = std::chrono::steady_clock::now();
                gloo::allreduce(opts.opts3);
                Count++;
                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                if(Count%300==0){
                        std::cout << opts.opts2.getImpl().elements << " elements2 "<< "Allreduce3 took " << duration.count() << " milliseconds." << std::endl;}
            });
            std::thread allreduceThread([&opts]() {
                auto start = std::chrono::steady_clock::now();
                gloo::allreduce(opts.opts2);
                Count2++;
                auto end = std::chrono::steady_clock::now();
                auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                if(Count2%300==0){
                        std::cout << opts.opts2.getImpl().elements << " elements2 "<< "Allreduce2 took " << duration2.count() << " milliseconds." << std::endl;      
                }          
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

