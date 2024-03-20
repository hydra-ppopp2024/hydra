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

class SPipeAllreduceOptions {
 public:
  explicit SPipeAllreduceOptions(const std::shared_ptr<Context>& context,const std::shared_ptr<Context>& context2, const char *ibname)
      : context(context),
        rank(context->rank),
        size(context->size),
        opts1(context, ibname),
        opts2(context2){
        }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    size_t elements1;
    size_t elements2;
    if(getenv("SHARP_GLEX") != NULL)
        calculateElements_SG(elements, &elements1, &elements2);
    else
        calculateElements_SA(elements, &elements1, &elements2);


    T* ptr1= ptr; 
    T* ptr2= ptr+elements1;

    //new add 
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
        if(elements2==0){
          this->opts1.setInput(ptr1, elements1);  
          } 
        else if(elements1==0){
          this->opts2.setInput(ptr2, elements2);
        }
        else{  
          this->opts1.setInput(ptr1, elements1);        
          this->opts2.setInput(ptr2, elements2); 
        }   
    }

  }


  using Func = detail::AllreduceOptionsImpl::Func;
  void setReduceFunction(Func fn) {
    this->opts2.setReduceFunction(fn);
//     this->opts3.setReduceFunction(fn);    
  }

  using Algorithm = detail::AllreduceOptionsImpl::Algorithm;
  void setAlgorithm(Algorithm algorithm) {
    this->opts2.setAlgorithm(algorithm);
//     this->opts3.setAlgorithm(algorithm);    
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
    else
        calculateElements_SA(elements, &elements1, &elements2);

    T* ptr1= ptr;
    T* ptr2= ptr+elements1;    
    if(getenv("SHARP_GLEX") != NULL || getenv("SHARP_ALLREDUCE") != NULL){
        opts1.elements=elements1;
        opts2.getImpl().elements=elements2;
        if(elements2==0){
          this->opts1.setOutput(ptr1, elements1);   
          }
        else if(elements1==0){
          this->opts2.setOutput(ptr2, elements2);   
          }
        else{      
          this->opts1.setOutput(ptr1, elements1);    
          this->opts2.setOutput(ptr2, elements2);
        }
    }
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
        else if(4096<elements && elements<262145){
                cout_ele=1;                
                w_2=1;
        }     
        else if(262144<elements && elements<524289){
                w_2=80;
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
                cout_ele=100;
                w_2=74;             // 72->74
        }           
        else if(1048576<elements && elements<2097153){
                cout_ele=100;
                w_2=72;  //70->72
        }            
        else if(2097152<elements && elements<4194305){
                cout_ele=100;
                w_2=70;            //68->70
        }    
        else if(4194304<elements && elements<8388609){    
                cout_ele=100;
                w_2=67;       //65->67
        }                       
        else if(8388608<elements && elements<16777217){ 
                cout_ele=100;
                w_2=64;         //62->64
        }          
        else if(16777216<elements && elements<33554433){
                cout_ele=100;
                w_2=62;         //60->62
        }     
        else if(33554432<elements && elements<67108865){
                cout_ele=100;
                w_2=60;       //58->60
        }      
        else{
                cout_ele=1;
                w_2=1;
        } 
    }
    else if(this->context->size==4){
        cout_ele=100;
        w_2=50;        
        /*
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
                w_2=72;       //70->72
        }          
        else if(524288<elements && elements<1048577){
                w_2=72;                //70->72
        }           
        else if(1048576<elements && elements<2097153){
                w_2=70;               //69->70
        }            
        else if(2097152<elements && elements<4194305){
                w_2=71;             //70->71
        }    
        else if(4194304<elements && elements<8388609){    
                w_2=69;         //68->69
        }                       
        else if(8388608<elements && elements<16777217){
                w_2=62;       //61->62
        }          
        else if(16777216<elements && elements<33554433){ 
                w_2=62;       //60->62
        }     
        else if(33554432<elements && elements<67108865){
                w_2=60;       
        }         
        else{
                cout_ele=2;
                w_2=1;
        }
*/
    }
    else if(this->context->size==6){
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
                w_2=70;                
        }           
        else if(1048576<elements && elements<2097153){
                w_2=69;
        }            
        else if(2097152<elements && elements<4194305){
                w_2=70;             
        }    
        else if(4194304<elements && elements<8388609){    
                w_2=68;       
        }                       
        else if(8388608<elements && elements<16777217){
                w_2=61;       
        }          
        else if(16777216<elements && elements<33554433){ 
                w_2=60;       
        }     
        else if(33554432<elements && elements<67108865){
                w_2=60;       
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
                //     std::cout << w_2<< "w2--element"<<elements  << std::endl;
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
    else if(this->context->size==3){
                cout_ele=100;        
        if(elements<262145){
                cout_ele=1;
                w_2=0;
            }   
        else if(262144<elements && elements<524289){
                w_2=36;  //30->35->36
        }     
        else if(524288<elements && elements<1048577){
                w_2=46;   //43->44->46
        }  
        else if(1048576<elements && elements<2097153){
                w_2=47;    //45->47
        }          
        else if(2097152<elements && elements<4194305){
                w_2=48;        //45-> 47  ->48     
        }             
        else if(4194304<elements && elements<8388609){
                w_2=48;        //45-> 47   ->48 
        }       
        else if(8388608<elements && elements<16777217){
                w_2=48;        //45-> 47    ->48            
        }          
        else if(16777216<elements && elements<33554433){
                w_2=48;        //45-> 47  ->48                 
        }     
        else if(33554432<elements && elements<67108865){
                w_2=48;        //45-> 47           ->48     
        }         

        else{
                cout_ele=2;
                w_2=1;
        }

    }
    else if(this->context->size==4){
                cout_ele=100;
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
                w_2=22;                 //18->20->22
        }     
        else if(524288<elements && elements<1048577){
                w_2=42;                //38->40->42
        }  
        else if(1048576<elements && elements<2097153){
                w_2=46;             //45->46
        }                
        else if(2097152<elements && elements<4194305){
                w_2=46;            //44->46
        }    
        else if(4194304<elements && elements<8388609){
                w_2=49;            //48->49
        }                      
        else if(8388608<elements && elements<16777217){
                w_2=48;                
        }          
        else if(16777216<elements && elements<33554433){
                w_2=48;            
        }     
        else if(33554432<elements && elements<67108865){
                w_2=48;              
        }         

        else{
                cout_ele=2;
                w_2=1;
        }

    }
    else if(this->context->size==6){
                cout_ele=100;
        if(elements<524289){
                cout_ele=1;
                w_2=0;
            }    
        else if(524288<elements && elements<1048577){
                w_2=38;
        }  
        else if(1048576<elements && elements<2097153){
                w_2=42;    //45->42
        }                
        else if(2097152<elements && elements<4194305){
                w_2=46;            
        }    
        else if(4194304<elements && elements<8388609){
                w_2=47;            //48->47
        }                      
        else if(8388608<elements && elements<16777217){
                w_2=47;                //48->47
        }          
        else if(16777216<elements && elements<33554433){
                w_2=47;                //48->47     
        }     
        else if(33554432<elements && elements<67108865){
                w_2=47;                //48->47           
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
//   std::cout << w_2<< "w2--element---sa"<<elements  << std::endl;
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
//   AllreduceOptions opts3;
  struct Options {
    int ratio=0;
  } opts;

  friend void spipe_allreduce(SPipeAllreduceOptions&);
};

void spipe_allreduce(SPipeAllreduceOptions& opts);//gloo::pipe_allreduce(PipeAllreduceOptions);
// void getElements(size_t elements, int parac);
} // namespace gloo

