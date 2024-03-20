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
#include <unordered_map>
#include <iostream>
#include "gloo/allreduce.h"
#include "gloo/sharp_allreduce.h"
#include "sharp/api/sharp.h"
// #include "gloo/pipeallreduce.cc"



namespace gloo {

class APipeAllreduceOptions {
 public:
  explicit APipeAllreduceOptions(const std::shared_ptr<Context>& context,const std::shared_ptr<Context>& context2, const char *ibname)
      : context(context),
        rank(context->rank),
        size(context->size),
        opts2(context2),
        opts3(context){
        }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    size_t elements1;
    size_t elements2;
    if(getenv("ALLREDUCE_GLEX") != NULL) 
        calculateElements_AG(elements, &elements1, &elements2);
    else
        calculateElements_AA(elements, &elements1, &elements2);

    T* ptr1= ptr; 
    T* ptr2= ptr+elements1;
        opts3.getImpl().elements=elements1;
        opts2.getImpl().elements=elements2;
        // opts.elements=elements;
    //new add 
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
    }
    else{
        if(elements2==0){
          this->opts3.setInput(ptr1, elements1);  
          } 
        else if(elements1==0){
          this->opts2.setInput(ptr2, elements2);
        }
        else{  
          this->opts3.setInput(ptr1, elements1);        
          this->opts2.setInput(ptr2, elements2); 
        }         
    }
    // */

// this->opts2.setInput(ptr, elements);


  }


  using Func = detail::AllreduceOptionsImpl::Func;
  void setReduceFunction(Func fn) {
    this->opts2.setReduceFunction(fn);
    this->opts3.setReduceFunction(fn);    
  }

  using Algorithm = detail::AllreduceOptionsImpl::Algorithm;
  void setAlgorithm(Algorithm algorithm) {
    this->opts2.setAlgorithm(algorithm);
    this->opts3.setAlgorithm(algorithm);    
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
    if(getenv("ALLREDUCE_GLEX") != NULL) 
        calculateElements_AG(elements, &elements1, &elements2);
    else
        calculateElements_AA(elements, &elements1, &elements2);

    T* ptr1= ptr;
    T* ptr2= ptr+elements1;    
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
    }
    else{
        opts3.getImpl().elements=elements1;
        opts2.getImpl().elements=elements2;
        // opts.elements=elements;
        if(elements2==0){
          this->opts3.setOutput(ptr1, elements1);   
          }
        else if(elements1==0){
          this->opts2.setOutput(ptr2, elements2);   
          }
        else{      
          this->opts3.setOutput(ptr1, elements1);    
          this->opts2.setOutput(ptr2, elements2);
          // std::cout << "elements1 " << elements1<< "elements2 " << elements2<< std::endl; 
        }      
    }
// */          //   ALLREDUCE_GLEX ALLREDUCE_SHARP ALLREDUCE_ALLREDUCE_     
      // this->opts2.setOutput(ptr, elements);
  }


void calculateElements_AG(size_t elements, size_t* elements1, size_t* elements2) {
    int cout_ele;
    int cout_mode;
    int w_2;
    if(this->context->size==2){
        cout_ele=100;
        if(elements<262145){
                cout_ele=1;
                w_2=1;
            }       
        else if(262144<elements && elements<524289){
                cout_ele=1;
                w_2=1;
        }     
        else if(524288<elements && elements<1048577){
                w_2=75;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=74;
        }          
        else if(2097152<elements && elements<4194305){
                w_2=72;                
        }             
        else if(4194304<elements && elements<8388609){
                w_2=69;                
        }       
        else if(8388608<elements && elements<16777217){
                w_2=67;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=65;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=65;                
        }   
        else{
                w_2=60;
        }
    }
    else if(this->context->size==3){
                cout_ele=100;           
        if(elements<524289){
                cout_ele=1;
                w_2=1;
            }
        else if(524288<elements && elements<1048577){
                w_2=80;
        }          
        else if(1048576<elements && elements<2097153){
                cout_ele=15;
                w_2=11;
        }                   
        else if(2097152<elements && elements<4194305){
                w_2=70;                 //71->70
        }    
        else if(4194304<elements && elements<8388609){
                w_2=68;                //69->68
        }                       
        else if(8388608<elements && elements<16777217){
                w_2=64;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=65;                //64->66->65
        }     
        else if(8388608<elements && elements<67108865){
                w_2=64;                
        }         
        else{
                cout_ele=2;
                w_2=1;
        }
    }
    else if(this->context->size==4){
        cout_ele=100;        
        if(elements<828344){
                cout_ele=1;
                w_2=1;
            }
        else if(828343<elements && elements<1048577){
                w_2=81;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=73;
        }                   
        else if(2097152<elements && elements<4194305){
                w_2=70;                
        }    
        else if(4194304<elements && elements<8388609){
                w_2=67;        
        }                       
        else if(8388608<elements && elements<16777217){
                w_2=65;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=65;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=66;                
        }         
        else{
                cout_ele=2;
                w_2=1;
        }
    }
    else if(this->context->size==6){
        cout_ele=100;        
        if(elements<1048577){
                cout_ele=1;
                w_2=1;
            }
        else if(1048576<elements && elements<2097153){
                w_2=73;
        }                   
        else if(2097152<elements && elements<4194305){
                w_2=70;                
        }    
        else if(4194304<elements && elements<8388609){
                w_2=66;        //67->66
        }                       
        else if(8388608<elements && elements<16777217){
                w_2=66;         //65->66 tongyitry       
        }          
        else if(16777216<elements && elements<33554433){
                w_2=66;         //65->66 tongyitry        
        }     
        else if(33554432<elements && elements<67108865){
                w_2=66;                
        }         
        else{
                cout_ele=2;
                w_2=1;
        }
    }

    else{
        if(elements<6145){
                cout_ele=1;
                w_2=0;
            }
        else if(6144<elements && elements<114975){
                cout_ele=1;
                w_2=1;
        }       
        else{
                cout_ele=1;
                w_2=1;
        }  
    }
    cout_mode = elements % cout_ele;
    if (cout_mode == 0) {
            *elements2 = w_2 * elements / cout_ele;
            *elements1 = elements - *elements2;
    }
    else {
            *elements2 = w_2 * (elements - cout_mode) / cout_ele;
            *elements1 = elements - *elements2;
    }  
}

void calculateElements_AA(size_t elements, size_t* elements1, size_t* elements2) {
    int cout_ele;
    int cout_mode;
    int w_2;
    
    if(this->context->size==2){
        if(elements<65536){
                cout_ele=1;
                w_2=1;
            }
        else if(1048576<elements && elements<2097153){
                cout_ele=100;
                w_2=48;
        }           
        else{
                cout_ele=2;
                w_2=1;
        }

    }
    else if(this->context->size==3){
        if(elements<131072){
                cout_ele=1;
                w_2=1;
            }  
        else{
                cout_ele=2;
                w_2=1;
        } 
    }
    else if(this->context->size==4){
        if(elements<65537){
                cout_ele=1;
                w_2=1;
            }      
        else if(524287<elements && elements<16777217){
                cout_ele=100;
                w_2=52;
        }                
        else{
                cout_ele=2;
                w_2=1;
        }

    }
    else if(this->context->size==6){
        if(elements<65537){
                cout_ele=1;
                w_2=1;
            }      
        // else if(524287<elements && elements<16777217){
        //         cout_ele=100;
        //         w_2=52;
        // }                
        else{
                cout_ele=2;
                w_2=1;
        }

    }

    else{
        if(elements<131072){
                cout_ele=1;
                w_2=1;
            }
        else{
                cout_ele=2;
                w_2=1;
        }  
    }
    cout_mode = elements % cout_ele;
    if (cout_mode == 0) {
            *elements2 = w_2 * elements / cout_ele;
            *elements1 = elements - *elements2;
    }
    else {
            *elements2 = w_2 * (elements - cout_mode) / cout_ele;
            *elements1 = elements - *elements2;
    }  
}

  // int ratio=2;
  size_t elements = 256;
  size_t elementSize = 0;
  size_t parac=0;

private:
  int rank;
  int size;
  std::shared_ptr<Context> context;
  std::shared_ptr<Context> context2;  
  AllreduceOptions opts2;
//   const char *ibname;
  AllreduceOptions opts3;
  struct Options {
    int ratio=0;
    size_t elements;
  } opts;

  friend void apipe_allreduce(APipeAllreduceOptions&);
};

void apipe_allreduce(APipeAllreduceOptions& opts);//gloo::pipe_allreduce(PipeAllreduceOptions);
// void getElements(size_t elements, int parac);
} // namespace gloo

