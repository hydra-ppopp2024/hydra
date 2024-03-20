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

class PipeAllreduceOptions {
 public:
  explicit PipeAllreduceOptions(const std::shared_ptr<Context>& context,const std::shared_ptr<Context>& context2, const char *ibname)
      : context(context),
        rank(context->rank),
        size(context->size),
        opts1(context, ibname),
        opts2(context2),
        opts3(context){
        }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    size_t elements1;
    size_t elements2;
    if(getenv("SHARP_GLEX") != NULL)
        calculateElements_SG(elements, &elements1, &elements2);
    else if(getenv("SHARP_ALLREDUCE") != NULL) 
        calculateElements_SA(elements, &elements1, &elements2);
    else if(getenv("ALLREDUCE_GLEX") != NULL) 
        calculateElements_AG(elements, &elements1, &elements2);
    else
        calculateElements_AA(elements, &elements1, &elements2);

    T* ptr1= ptr; 
    T* ptr2= ptr+elements1;

    //new add 
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
        // if(elements2==0){
        //   this->opts1.setInput(ptr1, elements1);  
        //   } 
        // else if(elements1==0){
        //   this->opts2.setInput(ptr2, elements2);
        // }
        // else{  
        //   this->opts1.setInput(ptr1, elements1);        
        //   this->opts2.setInput(ptr2, elements2); 
        // }   
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
    if(getenv("SHARP_GLEX") != NULL)
        calculateElements_SG(elements, &elements1, &elements2);
    else if(getenv("SHARP_ALLREDUCE") != NULL) 
        calculateElements_SA(elements, &elements1, &elements2);
    else if(getenv("ALLREDUCE_GLEX") != NULL) 
        calculateElements_AG(elements, &elements1, &elements2);
    else
        calculateElements_AA(elements, &elements1, &elements2);

    T* ptr1= ptr;
    T* ptr2= ptr+elements1;    
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
        // opts1.elements=elements1;
        // opts2.getImpl().elements=elements2;
        // if(elements2==0){
        //   this->opts1.setOutput(ptr1, elements1);   
        //   }
        // else if(elements1==0){
        //   this->opts2.setOutput(ptr2, elements2);   
        //   }
        // else{      
        //   this->opts1.setOutput(ptr1, elements1);    
        //   this->opts2.setOutput(ptr2, elements2);
        // }
    }
    else{
        opts3.getImpl().elements=elements1;
        opts2.getImpl().elements=elements2;
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

void calculateElements_SG(size_t elements, size_t* elements1, size_t* elements2) {
    int cout_ele;
    int w_2;
    int cout_mode;    
    if(this->context->size==2){
        cout_ele=100;
        if(elements<4097){
                cout_ele=1;                
                w_2=0;
            }
        else if(4096<elements && elements<65537){
                cout_ele=1;                
                w_2=1;
        } 
        else if(65536<elements && elements<131073){
                w_2=83;
        }         
        else if(131072<elements && elements<262145){
                w_2=82;
        }        
        else if(262144<elements && elements<524289){
                w_2=74;
        }     
        else if(524288<elements && elements<1048577){
                w_2=76;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=75;
        }          
        else if(2097152<elements && elements<4194305){
                w_2=74;                
        }             
        else if(4194304<elements && elements<8388609){
                w_2=72;                
        }       
        else if(8388608<elements && elements<16777217){
                w_2=65;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=64;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=65;                
        }   
        else{
                w_2=50;
        }

    }
    else if(this->context->size==3){
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
    else if(this->context->size==4){
        if(elements<14935){
                cout_ele=1;
                w_2=0;
            }
        else if(14934<elements && elements<65537){
                cout_ele=1;
                w_2=1;
        }  
        else if(65536<elements && elements<131073){
                cout_ele=13;
                w_2=11;
        }          
        else if(131072<elements && elements<262145){
                cout_ele=61;
                w_2=45;
        }                   
        else if(262144<elements && elements<524289){
                cout_ele=59;
                w_2=42;
        }          
        else if(524288<elements && elements<1048577){
                cout_ele=97;
                w_2=67;                
        }           
        else if(1048576<elements && elements<2097153){
                cout_ele=61;
                w_2=41;
        }            
        else if(2097152<elements && elements<4194305){
                cout_ele=67;
                w_2=45;             
        }    
        else if(4194304<elements && elements<8388609){    
                cout_ele=11;
                w_2=7;       
        }                       
        else if(8388608<elements && elements<16777217){
                // cout_ele=5;
                // w_2=3;    
                cout_ele=86;
                w_2=51;       
        }          
        else if(16777216<elements && elements<33554433){
                // cout_ele=86;
                // w_2=51;   
                cout_ele=84;
                w_2=47;       
        }     
        else if(33554432<elements && elements<67108865){
                // cout_ele=41;
                // w_2=23;  
                cout_ele=82;
                w_2=45;       
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

void calculateElements_SA(size_t elements, size_t* elements1, size_t* elements2) {
    int cout_ele;
    int w_2;
    int cout_mode;    
    if(this->context->size==2){
        cout_ele=100;
        if(elements<65537){
                w_2=0;
            }
        else if(65536<elements && elements<131073){
                w_2=28;
        } 
        else if(131072<elements && elements<262145){
                w_2=37;
        }        
        else if(262144<elements && elements<524289){
                w_2=45;
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
                w_2=43;                
        }       
        else if(8388608<elements && elements<16777217){
                w_2=41;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=42;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=44;                
        }   
        else{
                w_2=1;
        }

    }
    else if(this->context->size==3){
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
    else if(this->context->size==4){

        if(elements<347756){
                cout_ele=1;
                w_2=0;
            }
        else if(347755<elements && elements<380524){
                cout_ele=11;
                w_2=1;
        }        
        else if(380523<elements && elements<461208){
                cout_ele=7;
                w_2=1;
        }      
        else if(461207<elements && elements<524289){
                cout_ele=7;
                w_2=1;
        }     
        else if(524288<elements && elements<655361){
                cout_ele=5;
                w_2=1;
        }  
        else if(655360<elements && elements<828344){
                cout_ele=8;
                w_2=3;
        }  
        else if(828343<elements && elements<1048577){
                cout_ele=43;
                w_2=15;
        }  
        else if(1048576<elements && elements<2097153){
                cout_ele=67;
                w_2=27;
        }          
        else if(2097152<elements && elements<8388609){
                cout_ele=5;
                w_2=2;                
        }             
        else if(8388608<elements && elements<16777217){
                cout_ele=5;
                w_2=2;                
        }          
        else if(16777216<elements && elements<33554433){
                cout_ele=9;
                w_2=4;                
        }     
        else if(33554432<elements && elements<67108865){
                cout_ele=7;
                w_2=3;                
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
                w_2=89;
        }     
        else if(524288<elements && elements<1048577){
                w_2=80;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=76;
        }          
        else if(2097152<elements && elements<4194305){
                w_2=75;                
        }             
        else if(4194304<elements && elements<8388609){
                w_2=73;                
        }       
        else if(8388608<elements && elements<16777217){
                w_2=71;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=71;                
        }     
        else if(33554432<elements && elements<67108865){
                w_2=71;                
        }   
        else{
                w_2=50;
        }
    }
    else if(this->context->size==3){
        if(elements<262145){
                cout_ele=1;
                w_2=1;
            }
        else if(262144<elements && elements<524289){
                cout_ele=13;
                w_2=12;
        }  
        else if(524288<elements && elements<1048577){
                cout_ele=46;
                w_2=39;
        }          
        else if(1048576<elements && elements<2097153){
                cout_ele=15;
                w_2=11;
        }                   
        else if(2097152<elements && elements<4194305){
                cout_ele=7;
                w_2=5;                
        }    
        else if(4194304<elements && elements<8388609){
                cout_ele=73;
                w_2=51;                
        }                       
        else if(8388608<elements && elements<16777217){
                cout_ele=34;
                w_2=23;                
        }          
        else if(16777216<elements && elements<33554433){
                cout_ele=59;
                w_2=39;                
        }     
        else if(8388608<elements && elements<67108865){
                cout_ele=81;
                w_2=53;                
        }         
        else{
                cout_ele=2;
                w_2=1;
        }
    }
    else if(this->context->size==4){
        if(elements<828344){
                cout_ele=1;
                w_2=1;
            }
        else if(828343<elements && elements<1048577){
                cout_ele=10;
                w_2=9;
        }  
        else if(1048576<elements && elements<2097153){
                cout_ele=4;
                w_2=3;
        }                   
        else if(2097152<elements && elements<4194305){
                cout_ele=15;
                w_2=11;                
        }    
        else if(4194304<elements && elements<8388609){
                cout_ele=7;
                w_2=5;                
        }                       
        // else if(8388608<elements && elements<16777217){
        //         cout_ele=13;
        //         w_2=9;                
        // }          
        // else if(16777216<elements && elements<33554433){
        //         cout_ele=77;
        //         w_2=52;                
        // }     
        else if(8388608<elements && elements<67108865){
                cout_ele=50;
                w_2=31;                
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
    int cout_ele = 2;
    int cout_mode;
    int w_2;
    if(this->context->size==2){
        if(elements<6145){
                cout_ele=1;
                w_2=0;
            }
        else if(6144<elements && elements<114975){
                cout_ele=1;
                w_2=1;
        } 
        else if(114974<elements && elements<114975){
                cout_ele=1;
                w_2=1;
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
    else if(this->context->size==3){
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
    else if(this->context->size==4){
        if(elements<828344){
                cout_ele=1;
                w_2=1;
            }
        else if(828343<elements && elements<1048577){
                cout_ele=10;
                w_2=9;
        }  
        // else if(1048576<elements && elements<2097153){
        //         cout_ele=69;
        //         w_2=53;
        // }                   
        // else if(2097152<elements && elements<4194305){
        //         cout_ele=15;
        //         w_2=11;                
        // }    
        // else if(4194304<elements && elements<8388609){
        //         cout_ele=7;
        //         w_2=5;                
        // }                       
        // else if(8388608<elements && elements<16777217){
        //         cout_ele=13;
        //         w_2=9;                
        // }          
        // else if(16777216<elements && elements<33554433){
        //         cout_ele=77;
        //         w_2=52;                
        // }     
        else if(1048576<elements && elements<67108865){
                cout_ele=10;
                w_2=7;                
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

  // int ratio=2;
  size_t elements = 256;
  size_t elementSize = 0;
  size_t parac=0;

private:
  int rank;
  int size;
  std::shared_ptr<Context> context;
  std::shared_ptr<Context> context2;  
  SharpAllreduceOptions opts1;
  AllreduceOptions opts2;
//   const char *ibname;
  AllreduceOptions opts3;
  struct Options {
    int ratio=0;
  } opts;

  friend void pipe_allreduce(PipeAllreduceOptions&);
};

void pipe_allreduce(PipeAllreduceOptions& opts);//gloo::pipe_allreduce(PipeAllreduceOptions);
// void getElements(size_t elements, int parac);
} // namespace gloo

