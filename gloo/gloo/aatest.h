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
#include <cstring>
#include <unordered_map>
#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"
#include "gloo/barrier.h"
#include "gloo/broadcast.h"
#include "gloo/gather.h"

#include "gloo/allreduce.h"
#include "gloo/sharp_allreduce.h"
#include "sharp/api/sharp.h"



namespace gloo {

class PipeAllreduceOptions {
 public:
  explicit PipeAllreduceOptions(const std::shared_ptr<Context>& context,const std::shared_ptr<Context>& context2, const char *ibname)
      : context(context),
        rank(context->rank),
        size(context->size),
        opts1(context2, ibname),
        opts2(context){
        }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    size_t elements1;
    size_t elements2;
    bool USE_SHARP=true;
    //old
  // /*
       if(this->context->size==4 && getenv("SHARP_COLL_ENABLE_SAT=1")== NULL ){
            if(elements<16384){
                elements1=elements;//2是allreduce
                elements2=0;
            }
            else if(16383<elements<49153){
                elements1=0;//2是allreduce
                elements2=elements;
            }            
            else if(49152<elements<8388609 ){//6 or5?
                int cout_ele=6;
                int cout_mode;
                cout_mode=elements%cout_ele;
                if(cout_mode==0){
                    elements1=5*elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                }
                else{
                    elements1=5*(elements-cout_mode)/cout_ele;
                    elements2=elements-elements1;
                    }   
            }
            else if(8388608<elements<67108865 ){
                    int cout_ele=3;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements1=2*elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                  }
                  else{
                    elements1=2*(elements-cout_mode)/cout_ele;
                    elements2=elements-elements1;
                      }   
            }
            else{
                    int cout_ele=17;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements1=7*elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                  }
                    else{
                      elements1=7*(elements-cout_mode)/cout_ele;
                      elements2=elements-elements1;
                        }   
            }
    }
    else{
                int cout_ele=2;
                int cout_mode;
                cout_mode=elements%cout_ele;
                if(cout_mode==0){
                    elements1=elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                }
                else{
                    elements1=(elements-cout_mode)/cout_ele;
                    elements2=elements-elements1;
                    }   
    }
    T* ptr1= ptr; 
    T* ptr2= ptr+elements1;
opts1.elements=elements1;
opts2.getImpl().elements=elements2;
    //new add 
    if(elements2==0){
      this->opts1.setInput(ptr1, elements1);   
      } 
      else if(elements1==0){
      this->opts2.setInput(ptr2, elements2);  
      } 
    else{  
      this->opts1.setInput(ptr1, elements1);    
      // std::cout << "ptr1 " << ptr1 << std::endl;
      // std::cout << "elements1 " << elements1<< std::endl;      
      this->opts2.setInput(ptr2, elements2);
      // std::cout << "ptr2 " << ptr2 << std::endl;
      // std::cout << "elements2 " << elements2<< std::endl;        
    }   
    // */

// this->opts2.setInput(ptr, elements);


  }


  using Func = detail::AllreduceOptionsImpl::Func;
  void setReduceFunction(Func fn) {
    this->opts2.setReduceFunction(fn);
  }

  using Algorithm = detail::AllreduceOptionsImpl::Algorithm;
  void setAlgorithm(Algorithm algorithm) {
    this->opts2.setAlgorithm(algorithm);
  }

#if GLOO_HAVE_TIMELINE
  void setTensorNames(const std::vector<std::string>& names){
    // impl_.tensor_names = names;
     this->opts2.setTensorNames(names);   
    // fprintf(stderr, "[gloo::allreduce] set tensor name:%s\n", impl_.tensor_name.c_str());
  }
#endif


  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    size_t elements1;
    size_t elements2;
    bool USE_SHARP=true;
  //  /*

       if(this->context->size==4 && getenv("SHARP_COLL_ENABLE_SAT=1")== NULL){ 
            if(elements<16384){
                elements1=elements;//2是allreduce
                elements2=0;
            }
            else if(16383<elements<49153){
                elements1=0;//2是allreduce
                elements2=elements;
            }            
            else if(49152<elements<8388609 ){//6 or5?
                int cout_ele=6;
                int cout_mode;
                cout_mode=elements%cout_ele;
                if(cout_mode==0){
                    elements1=elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                }
                else{
                    elements1=(elements-cout_mode)/cout_ele;
                    elements2=elements-elements1;
                    }   
            }
            // else if(1048576<elements<8388609 ){
            //         int cout_ele=6;
            //         int cout_mode;
            //         cout_mode=elements%cout_ele;
            //         if(cout_mode==0){
            //         elements1=5*elements/cout_ele;//2是allreduce
            //         elements2=elements-elements1;
            //       }
            //       else{
            //         elements1=5*(elements-cout_mode)/cout_ele;
            //         elements2=elements-elements1;
            //           }   
            // }
            else if(8388608<elements<67108865 ){
                    int cout_ele=3;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements1=elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                  }
                  else{
                    elements1=(elements-cout_mode)/cout_ele;
                    elements2=elements-elements1;
                      }   
            }

            else{
                    int cout_ele=17;
                    int cout_mode;
                    cout_mode=elements%cout_ele;
                    if(cout_mode==0){
                    elements1=7*elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                  }
                    else{
                      elements1=7*(elements-cout_mode)/cout_ele;
                      elements2=elements-elements1;
                        }   
            }
    }
        else{
                int cout_ele=2;
                int cout_mode;
                cout_mode=elements%cout_ele;
                if(cout_mode==0){
                    elements1=elements/cout_ele;//2是allreduce
                    elements2=elements-elements1;
                }
                else{
                    elements1=(elements-cout_mode)/cout_ele;
                    elements2=elements-elements1;
                    }   
    }
opts1.elements=elements1;
opts2.getImpl().elements=elements2;

    T* ptr1= ptr;
    T* ptr2= ptr+elements1;    
    if(elements2==0){

      this->opts1.setOutput(ptr1, elements1);  
      } 
      else if(elements1==0){
      this->opts2.setOutput(ptr2, elements2);  
      } 
    else{      
      this->opts1.setOutput(ptr1, elements1);    
      // std::cout << "ptr1 " << ptr1 << std::endl;
      // std::cout << "elements1 " << elements1<< std::endl;   
      this->opts2.setOutput(ptr2, elements2);

       //std::cout << "ratio.hhh" << setout() << std::endl;
      // std::cout << "elements2 " << elements2<< std::endl;         
    }
// */
      // this->opts2.setOutput(ptr, elements);
  }

  size_t elements = 256;
  size_t elementSize = 0;
private:
  int rank;
  int size;
  std::shared_ptr<Context> context;
  std::shared_ptr<Context> context2;  
  SharpAllreduceOptions opts1;
  AllreduceOptions opts2;



  friend void pipe_allreduce(PipeAllreduceOptions&);
};

void pipe_allreduce(PipeAllreduceOptions& opts);//gloo::pipe_allreduce(PipeAllreduceOptions);

} // namespace gloo

