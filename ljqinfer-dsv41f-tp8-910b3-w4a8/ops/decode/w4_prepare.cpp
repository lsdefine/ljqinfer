#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
template<HardEvent E> __aicore__ inline void wait_event(){SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
// Restore HP-mutated quantized scratch and recompute all group counts.
// No floating point arithmetic, allocations, or host reads of group metadata.
__global__ __aicore__ void prepare_input(GM_ADDR raw,GM_ADDR q,GM_ADDR ends,GM_ADDR counts,
                                        int rows,int k,int kp,int experts,int blocks){
    TPipe pipe;TBuf<TPosition::VECCALC> data,meta,out;
    pipe.InitBuffer(data,5120);pipe.InitBuffer(meta,384*8);pipe.InitBuffer(out,16*8);
    auto bytes=data.Get<int8_t>();auto zeros=data.Get<uint16_t>();
    auto ee=meta.Get<int64_t>();auto cc=out.Get<int64_t>();
    GlobalTensor<int8_t> gr,gq;gr.SetGlobalBuffer((__gm__ int8_t*)raw);gq.SetGlobalBuffer((__gm__ int8_t*)q);
    GlobalTensor<int64_t> ge,gc;ge.SetGlobalBuffer((__gm__ int64_t*)ends);gc.SetGlobalBuffer((__gm__ int64_t*)counts);
    for(int task=GetBlockIdx();task<rows+experts/16;task+=blocks){
        if(task<rows){
            if(kp>k){Duplicate(zeros,(uint16_t)0,kp/2);wait_event<HardEvent::V_MTE2>();}
            DataCopy(bytes,gr[task*k],k);wait_event<HardEvent::MTE2_MTE3>();
            DataCopy(gq[task*kp],bytes,kp);wait_event<HardEvent::MTE3_V>();wait_event<HardEvent::MTE3_MTE2>();
        }else{
            int first=(task-rows)*16;
            DataCopy(ee,ge,experts);wait_event<HardEvent::MTE2_S>();
            for(int j=0;j<16;++j){int at=first+j;cc.SetValue(j,ee.GetValue(at)-(at?ee.GetValue(at-1):0));}
            wait_event<HardEvent::S_MTE3>();DataCopy(gc[first],cc,16);wait_event<HardEvent::MTE3_S>();
        }
    }
}
extern "C" int dec_w4_prepare(void* stream,void* raw,void* q,void* ends,void* counts,int rows,int k,int experts){
    if(!stream||!raw||!q||!ends||!counts||rows<1||rows>144||(k!=288&&k!=5120)||(experts!=128&&experts!=384))return -1;
    if(((uintptr_t)raw|(uintptr_t)q|(uintptr_t)ends|(uintptr_t)counts)&31)return -1;
    int kp=(k+63)/64*64;int blocks=rows+experts/16;if(blocks>80)blocks=80;
    prepare_input<<<blocks,nullptr,stream>>>((uint8_t*)raw,(uint8_t*)q,(uint8_t*)ends,(uint8_t*)counts,rows,k,kp,experts,blocks);return 0;
}
