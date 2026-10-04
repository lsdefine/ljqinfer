// Decode-only BnQ6 primitives. Tensor buffers/stream are owned by caller.
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t Q=6,D=5120,M=8;
// meta[B,8]: slot,start,active,reserved. result: slot,start,accepted,next,active,error,0,0.
// Input copies avoid scalar GM-cache reuse across captured replays.
// error: 1 = invalid metadata/cursor, 2 = invalid greedy/input token.
extern "C" __global__ __aicore__ void accept_kernel(GM_ADDR I,GM_ADDR G,GM_ADDR MTA,GM_ADDR R,uint32_t B,uint32_t slots){
 if(GetBlockIdx()>=GetBlockNum())return;
 GlobalTensor<int64_t> ids,g,meta,res;
 ids.SetGlobalBuffer((__gm__ int64_t*)I);g.SetGlobalBuffer((__gm__ int64_t*)G);
 meta.SetGlobalBuffer((__gm__ int64_t*)MTA);res.SetGlobalBuffer((__gm__ int64_t*)R);
 TPipe pipe;TBuf<TPosition::VECCALC> bm,bi,bg;
 pipe.InitBuffer(bm,64);pipe.InitBuffer(bi,64);pipe.InitBuffer(bg,64);
 auto lm=bm.Get<int64_t>(),li=bi.Get<int64_t>(),lg=bg.Get<int64_t>();
 TQue<QuePosition::VECOUT,1> queue;pipe.InitBuffer(queue,1,64);
 for(uint32_t b=GetBlockIdx();b<B;b+=GetBlockNum()){
  DataCopy(lm,meta[b*M],int32_t(M));
  DataCopyPad(li,ids[b*Q],DataCopyExtParams{1,Q*8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
  DataCopyPad(lg,g[b*Q],DataCopyExtParams{1,Q*8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  int64_t slot=lm.GetValue(0),start=lm.GetValue(1),active=lm.GetValue(2);
  int64_t err=active&&(slot<0||slot>=slots||start<0||start>INT64_MAX-Q),a=0,t=-1;
  if(active&&!err){
   for(uint32_t q=0;q<Q;q++)
    if(li.GetValue(q)<0||li.GetValue(q)>=129280||lg.GetValue(q)<0||lg.GetValue(q)>=129280)err=2;
   if(!err){a=1;for(uint32_t q=1;q<Q;q++){if(li.GetValue(q)!=lg.GetValue(q-1))break;a++;}t=lg.GetValue(a-1);}
  }
  auto out=queue.AllocTensor<int64_t>();
  out.SetValue(0,slot);out.SetValue(1,start);out.SetValue(2,a);out.SetValue(3,t);
  out.SetValue(4,active);out.SetValue(5,err);out.SetValue(6,0);out.SetValue(7,0);
  queue.EnQue(out);out=queue.DeQue<int64_t>();DataCopy(res[b*M],out,M);queue.FreeTensor(out);
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 }
}
// BF16 embedding table[vocab,5120] -> hidden[B,6,4,5120]. Inactive rows are zero.
extern "C" __global__ __aicore__ void embed_kernel(GM_ADDR I,GM_ADDR MTA,GM_ADDR W,GM_ADDR H,uint32_t B,uint32_t vocab,uint32_t offset){
 if(GetBlockIdx()>=GetBlockNum())return;
 GlobalTensor<int64_t> ids,meta;GlobalTensor<uint16_t> w,h;
 ids.SetGlobalBuffer((__gm__ int64_t*)I);meta.SetGlobalBuffer((__gm__ int64_t*)MTA);
 w.SetGlobalBuffer((__gm__ uint16_t*)W);h.SetGlobalBuffer((__gm__ uint16_t*)H);
 TPipe pipe;TQue<QuePosition::VECIN,1> queue;pipe.InitBuffer(queue,1,D*2);
 for(uint32_t r=GetBlockIdx();r<B*Q;r+=GetBlockNum()){
  int64_t id=ids.GetValue(r)-int64_t(offset);auto v=queue.AllocTensor<uint16_t>();
  if(meta.GetValue((r/Q)*M+2)&&id>=0&&id<vocab)DataCopy(v,w[(uint64_t)id*D],D);
  else {Duplicate(v,(uint16_t)0,D);PipeBarrier<PIPE_ALL>();}
  queue.EnQue(v);v=queue.DeQue<uint16_t>();
  SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
  for(uint32_t j=0;j<4;j++)DataCopy(h[((uint64_t)r*4+j)*D],v,D);
  PipeBarrier<PIPE_ALL>();queue.FreeTensor(v);
 }
}
extern "C" int dec_accept(void* s,void* i,void* g,void* m,void* r,uint32_t b,uint32_t slots){
 if(!s||!i||!g||!m||!r||!b||b>64||!slots)return -1;
 accept_kernel<<<b<40?b:40,nullptr,s>>>((uint8_t*)i,(uint8_t*)g,(uint8_t*)m,(uint8_t*)r,b,slots);return 0;
}
extern "C" int dec_embed(void* s,void* i,void* m,void* w,void* h,uint32_t b,uint32_t vocab){
 if(!s||!i||!m||!w||!h||!b||b>64||!vocab)return -1;
 embed_kernel<<<b*Q<40?b*Q:40,nullptr,s>>>((uint8_t*)i,(uint8_t*)m,(uint8_t*)w,(uint8_t*)h,b,vocab,0);return 0;
}

// Each TP rank reads only its contiguous vocabulary shard; other IDs yield zero.
extern "C" int dec_embed_shard(void* s,void* i,void* m,void* w,void* h,
    uint32_t b,uint32_t rows,uint32_t offset){
 if(!s||!i||!m||!w||!h||!b||b>64||!rows)return -1;
 embed_kernel<<<b*Q<40?b*Q:40,nullptr,s>>>((uint8_t*)i,(uint8_t*)m,
     (uint8_t*)w,(uint8_t*)h,b,rows,offset);return 0;
}

// SWA-only rotary phases and initial HC PRE. inv_freq[32] is a cold constant.
extern "C" __global__ __aicore__ void window_prepare_kernel(
    GM_ADDR MTA,GM_ADDR INV,GM_ADDR FREQ,GM_ADDR PRE,uint32_t B){
 if(GetBlockIdx()>=GetBlockNum())return;
 GlobalTensor<int64_t> meta;GlobalTensor<float> inv,freq,pre;
 meta.SetGlobalBuffer((__gm__ int64_t*)MTA);
 inv.SetGlobalBuffer((__gm__ float*)INV);
 freq.SetGlobalBuffer((__gm__ float*)FREQ);
 pre.SetGlobalBuffer((__gm__ float*)PRE);
 TPipe pipe;TBuf<TPosition::VECCALC> bi,ba,bc,bs,bo,bp;
 pipe.InitBuffer(bi,128);pipe.InitBuffer(ba,128);pipe.InitBuffer(bc,128);
 pipe.InitBuffer(bs,128);pipe.InitBuffer(bo,256);pipe.InitBuffer(bp,32);
 auto i=bi.Get<float>();auto a=ba.Get<float>();auto c=bc.Get<float>();
 auto s=bs.Get<float>();auto o=bo.Get<float>();auto p=bp.Get<float>();
 DataCopy(i,inv,32);PipeBarrier<PIPE_ALL>();
 for(uint32_t r=GetBlockIdx();r<B*Q;r+=GetBlockNum()){
  bool active=meta.GetValue((r/Q)*M+2)!=0;
  float pos=active?float(meta.GetValue((r/Q)*M+1)+r%Q):0.0f;
  Muls(a,i,pos,32);PipeBarrier<PIPE_ALL>();
  Cos(c,a,32);Sin(s,a,32);PipeBarrier<PIPE_ALL>();
  for(uint32_t j=0;j<32;j++){
   o.SetValue(2*j,c.GetValue(j));o.SetValue(2*j+1,s.GetValue(j));
  }
  for(uint32_t j=0;j<8;j++)p.SetValue(j,active&&j==0?1.0f:0.0f);
  PipeBarrier<PIPE_ALL>();
  DataCopy(freq[uint64_t(r)*64],o,64);DataCopy(pre[uint64_t(r)*8],p,8);
  PipeBarrier<PIPE_ALL>();
 }
}
extern "C" int dec_window_prepare(void* stream,void* meta,void* inv,
    void* freqs,void* pre,uint32_t batch){
 if(!stream||!meta||!inv||!freqs||!pre||!batch||batch>64||freqs==pre)return -1;
 window_prepare_kernel<<<batch*6<40?batch*6:40,nullptr,stream>>>(
   (uint8_t*)meta,(uint8_t*)inv,(uint8_t*)freqs,(uint8_t*)pre,batch);
 return 0;
}

// One layer: pending BF16[B,6,512], result int64[B,8] from dec_accept (read-only),
// history BF16[slots,pad+ring,512] = WindowPast.main_kv. All buffers disjoint.
// Publish only accepted KV: never touch scratch [0,pad), pos, carry, or replay flags.
// Caller leases fully replayed slots, stages pending WITHOUT writing history, and
// orders all producers/consumers on the supplied CURRENT stream (warm before capture).
// Invalid/inactive/error/zero-count rows and all active duplicate slots are no-ops.
extern "C" __global__ __aicore__ void pending_ring_kernel(
    GM_ADDR P,GM_ADDR R,GM_ADDR H,uint32_t B,uint32_t slots,uint32_t ring,uint32_t pad){
 if(GetBlockIdx()>=GetBlockNum())return;
 GlobalTensor<uint16_t> pending,history;GlobalTensor<int64_t> result;
 pending.SetGlobalBuffer((__gm__ uint16_t*)P);history.SetGlobalBuffer((__gm__ uint16_t*)H);
 result.SetGlobalBuffer((__gm__ int64_t*)R);
 TPipe pipe;TQue<QuePosition::VECIN,1> queue;pipe.InitBuffer(queue,1,512*2);
 for(uint32_t b=GetBlockIdx();b<B;b+=GetBlockNum()){
  int64_t slot=result.GetValue(b*M),start=result.GetValue(b*M+1),n=result.GetValue(b*M+2);
  if(!result.GetValue(b*M+4)||result.GetValue(b*M+5)||slot<0||slot>=slots||
     start<0||n<1||n>Q||start>INT64_MAX-n)continue;
  bool conflict=false;
  for(uint32_t j=0;j<B;j++)
   if(j!=b&&result.GetValue(j*M+4)&&result.GetValue(j*M)==slot)conflict=true;
  if(conflict)continue;
  for(uint32_t q=0;q<uint32_t(n);q++){
   auto row=queue.AllocTensor<uint16_t>();
   DataCopy(row,pending[(uint64_t(b)*Q+q)*512],512);
   queue.EnQue(row);row=queue.DeQue<uint16_t>();
   SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
   uint64_t dst=(uint64_t(slot)*(uint64_t(pad)+ring)+pad+(uint64_t(start)+q)%ring)*512;
   DataCopy(history[dst],row,512);
   PipeBarrier<PIPE_ALL>();queue.FreeTensor(row);
  }
 }
}
extern "C" int dec_pending_ring(void* stream,void* pending,void* result,void* history,
    uint32_t batch,uint32_t slots,uint32_t ring,uint32_t pad){
 if(!stream||!pending||!result||!history||!batch||batch>64||!slots||ring<144||
    pad<16||uint64_t(pad)+ring>UINT32_MAX)return -1;
 uint64_t rows=uint64_t(slots)*(uint64_t(pad)+ring);
 if(rows>UINT64_MAX/1024)return -1;
 // Check complete byte ranges, not just equal base pointers. No host tensor reads.
 uintptr_t p=uintptr_t(pending),r=uintptr_t(result),h=uintptr_t(history);
 uint64_t pn=uint64_t(batch)*Q*1024,rn=uint64_t(batch)*M*8,hn=rows*1024;
 if(p>UINTPTR_MAX-pn||r>UINTPTR_MAX-rn||h>UINTPTR_MAX-hn||
    (p<r+rn&&r<p+pn)||(p<h+hn&&h<p+pn)||(r<h+hn&&h<r+rn))return -1;
 pending_ring_kernel<<<batch<40?batch:40,nullptr,stream>>>(
   (uint8_t*)pending,(uint8_t*)result,(uint8_t*)history,batch,slots,ring,pad);
 return 0;
}

// ---------------------------------------------------------------------------
// Fixed-resource TP8 greedy, ABI v1. No prefill, allocator, stream lookup,
// synchronization, communicator construction, or hidden persistent state.
// Canonical checkpoint: vocab=129280, rank r owns [r*16160,(r+1)*16160).
// Matches S:/ljqinfer_dsv41f_tp8/ops/decode/argmax.py: numeric FP32 (value,id)
// pairs, rank-major AllGather, equal values choose the LOWEST GLOBAL token id.
// IDs are exact numeric floats, NOT bit-casts. +/-inf and signed zero are valid;
// any NaN poisons its row (greedy=-1), so accept fails closed with error=2.
// Unlike the reference's -inf sentinel corner case, all -inf selects token 0.
//
// Caller owns contiguous, mutually disjoint, >=32-byte-aligned NPU buffers:
// logits  FP32 [B*6,16160]       (NativeHead.logits, read-only)
// partial FP32 [B*6,32,8]        (scratch: first 2 lanes value/id, rest zero)
// pair    FP32 [B*6,2]           (local send, exactly B*12 float32 elements)
// gathered FP32 [8,B*6,2]        (rank-major receive; no full-vocab gather)
// greedy  INT64 [B,6]            (existing dec_accept input)
// ids/meta/result are the existing INT64 [B,6]/[B,8]/[B,8] contract.
// B=1..64 for centralized capacity expansion; current model uses B=1..4.
// Bytes: partial=B*6144, pair=B*48, gathered=B*384, greedy=B*48.
//
// Typical native-only scheduling after head, all on SAME CURRENT stream:
//   dec_greedy_local(s, logits, partial, pair, B, rank);
//   HcclAllGather(pair, gathered, B*12, /*FP32=*/4, comm, s);
//   dec_greedy_accept(s, gathered, ids, meta, greedy, result, B, slots);
// Or dec_greedy_tp8(...) performs the first three steps through global select,
// followed by existing dec_accept(...). Raw HcclAllGather function pointer and
// comm are borrowed at startup (no prefill import required). world MUST be 8,
// communicator rank MUST equal the weight rank passed here. All ranks must use
// identical B and collective order, even when all local meta rows are inactive.
// Validate/warm communicator on every rank BEFORE capture. The caller enforces
// device/context, capacities, and cross-rank agreement (not discoverable here).
// TASK_QUEUE_ENABLE=0 when mixing raw launch/HCCL with torch_npu producers.
//
// Every invocation gets torch.npu.current_stream(device).npu_stream AT CALL
// TIME. Replays freeze pointers/scalars: never rebind storage, rank, B or comm.
// Keep buffers, library, communicator and HCCL library/function pointer alive
// until all dependent graphs are destroyed AND streams drained. No simultaneous
// replays may share scratch. These entries enqueue only; rc=0 is NOT device
// completion. On HCCL error, do not accept/commit and abort the TP round on ALL
// ranks. Host rc=-1 means bad pointers/scalars/alignment, -2 means overlap;
// positive rc from dec_greedy_tp8 is the unmodified HCCL status.
constexpr uint32_t GV=16160,GWORLD=8,GSPLIT=32,GCHUNK=512,GLANES=8;

// One independent 512-token tile per task. The final tile reads only 288
// tokens, never the next row. Explicit increasing-id scan makes ties, NaNs,
// +/-inf independent of hardware ReduceMax index/NaN conventions.
// DMA/UB only: no scalar GM loads/stores and no shared cache-line writers.
extern "C" __global__ __aicore__ void greedy_part_kernel(
    GM_ADDR X,GM_ADDR P,uint32_t rows,uint32_t rank,uint32_t blocks,uint32_t row_stride,uint32_t row_offset){
 if(GetBlockIdx()>=blocks)return;
 GlobalTensor<float> x,p;x.SetGlobalBuffer((__gm__ float*)X);p.SetGlobalBuffer((__gm__ float*)P);
 TPipe pipe;TBuf<TPosition::VECCALC> bx;
 pipe.InitBuffer(bx,GCHUNK*4);auto v=bx.Get<float>();
 TQue<QuePosition::VECOUT,1> qo;pipe.InitBuffer(qo,1,GLANES*4);
 for(uint32_t task=GetBlockIdx();task<rows*GSPLIT;task+=blocks){
  uint32_t row=task/GSPLIT,begin=(task%GSPLIT)*GCHUNK;
  uint32_t n=GV-begin<GCHUNK?GV-begin:GCHUNK;
  DataCopy(v,x[(uint64_t(row)*row_stride+row_offset)*GV+begin],int32_t(n));
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  float best=v.GetValue(0);uint32_t id=rank*GV+begin;bool bad=best!=best;
  for(uint32_t j=1;j<n;j++){
   float f=v.GetValue(j);bad=bad||(f!=f);
   if(f>best){best=f;id=rank*GV+begin+j;}
  }
  auto out=qo.AllocTensor<float>();
  out.SetValue(0,bad?0.0f:best);out.SetValue(1,bad?-1.0f:float(int32_t(id)));
  for(uint32_t j=2;j<GLANES;j++)out.SetValue(j,0.0f);
  qo.EnQue(out);out=qo.DeQue<float>();DataCopy(p[uint64_t(task)*GLANES],out,int32_t(GLANES));qo.FreeTensor(out);
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 }
}

// ---------------------------------------------------------------------------
// Temperature sampling shares the greedy reduction: with g=-log(-log u) for
// u~U(0,1), argmax(x + T*g) draws exactly from softmax(x/T), and T==0 is the
// plain argmax, so the split/join/AllGather/global stages are untouched and a
// sampled row costs one extra scalar draw per candidate. The draw is addressed
// in GLOBAL vocabulary space (row, rank*GV+j), so the eight sharded scans
// reproduce the single-device draw term for term. Matches the reference
// S:/ljqinfer_dsv41f_tp8/ops/decode/argmax.py sample_rows().
constexpr uint32_t GVFULL=GV*GWORLD;
__aicore__ inline uint32_t sample_mix(uint32_t k){
 k*=2654435761u;k^=k>>15;k*=2246822519u;k^=k>>13;k*=3266489917u;k^=k>>16;return k;
}
// Counter-based draw: no state, so any block may compute any (row, token).
__aicore__ inline float sample_uniform(uint32_t seed,uint32_t gid){
 uint32_t h=sample_mix(gid^(seed*2654435761u+0x9E3779B9u));
 h=sample_mix(h+0x85EBCA6Bu);
 // (h>>8)*2^-24 lands in [0,1); the half-ulp shift keeps log() finite.
 // AICore forbids unsigned<->float casts; 24 bits fit a signed int exactly.
 int32_t m=int32_t(h>>8);
 return float(m)*5.9604645e-8f+2.9802322e-8f;
}
// Natural log for x in (0,1]: no libm on the scalar unit, and the Gumbel term
// only needs ~1e-6, far below the noise it injects.
__aicore__ inline float sample_log(float x){
 union{float f;uint32_t u;}b;b.f=x;
 int32_t e=int32_t(b.u>>23)-127;
 b.u=(b.u&0x007FFFFFu)|0x3F800000u;
 float m=b.f;
 if(m>1.4142135f){m*=0.5f;e+=1;}
 float t=(m-1.0f)/(m+1.0f),t2=t*t;
 float s=t*(2.0f+t2*(0.6666667f+t2*(0.4f+t2*(0.2857143f+t2*0.2222222f))));
 return s+float(e)*0.6931472f;
}
__aicore__ inline float sample_gumbel(uint32_t seed,uint32_t gid){
 // u is strictly inside (0,1), yet the polynomial can still return a tiny
 // positive value for u just below 1; the second log would then explode and
 // hand the row to an arbitrary token, so clamp to a finite 16.1 ceiling.
 float e=-sample_log(sample_uniform(seed,gid));
 if(!(e>1e-7f))e=1e-7f;
 return -sample_log(e);
}
// Same contract as greedy_part_kernel plus TEMP FP32[rows/span] (one entry per
// request, broadcast over its MTP window) and SEED INT32[1].
extern "C" __global__ __aicore__ void sample_part_kernel(
    GM_ADDR X,GM_ADDR P,GM_ADDR T,GM_ADDR S,uint32_t rows,uint32_t rank,
    uint32_t blocks,uint32_t span){
 if(GetBlockIdx()>=blocks)return;
 GlobalTensor<float> x,p,tg;x.SetGlobalBuffer((__gm__ float*)X);
 p.SetGlobalBuffer((__gm__ float*)P);tg.SetGlobalBuffer((__gm__ float*)T);
 GlobalTensor<int32_t> sg;sg.SetGlobalBuffer((__gm__ int32_t*)S);
 TPipe pipe;TBuf<TPosition::VECCALC> bx,bt,bs;
 pipe.InitBuffer(bx,GCHUNK*4);auto v=bx.Get<float>();
 pipe.InitBuffer(bt,128);auto tv=bt.Get<float>();
 pipe.InitBuffer(bs,32);auto sv=bs.Get<int32_t>();
 TQue<QuePosition::VECOUT,1> qo;pipe.InitBuffer(qo,1,GLANES*4);
 DataCopyPad(sv,sg,DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 uint32_t seed=uint32_t(sv.GetValue(0));
 SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 // One request count is at most a handful of floats: load every temperature
 // once, so the hot loop holds no MTE2 traffic that must race the scalar
 // reads of the logits tile.
 uint32_t nreq=(rows+span-1)/span;
 DataCopyPad(tv,tg,DataCopyExtParams{1,nreq*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 for(uint32_t task=GetBlockIdx();task<rows*GSPLIT;task+=blocks){
  uint32_t row=task/GSPLIT,begin=(task%GSPLIT)*GCHUNK;
  uint32_t n=GV-begin<GCHUNK?GV-begin:GCHUNK;
  DataCopy(v,x[uint64_t(row)*GV+begin],int32_t(n));
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  float temp=tv.GetValue(row/span);
  // A non-positive, NaN or absurd temperature degrades to greedy, never to a
  // poisoned row: sampling must not be able to break a served request.
  bool draw=temp>0.0f&&temp<1.0e4f;
  uint32_t base=row*GVFULL+rank*GV+begin;
  float best=v.GetValue(0);
  bool bad=best!=best;
  if(draw&&!bad)best+=temp*sample_gumbel(seed,base);
  uint32_t id=rank*GV+begin;
  for(uint32_t j=1;j<n;j++){
   float f=v.GetValue(j);bad=bad||(f!=f);
   if(draw&&f==f)f+=temp*sample_gumbel(seed,base+j);
   if(f>best){best=f;id=rank*GV+begin+j;}
  }
  auto out=qo.AllocTensor<float>();
  out.SetValue(0,bad?0.0f:best);out.SetValue(1,bad?-1.0f:float(int32_t(id)));
  for(uint32_t j=2;j<GLANES;j++)out.SetValue(j,0.0f);
  qo.EnQue(out);out=qo.DeQue<float>();DataCopy(p[uint64_t(task)*GLANES],out,int32_t(GLANES));qo.FreeTensor(out);
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 }
}
// Advances the captured-graph draw. Every rank holds the same counter and adds
// the same 1, so the ranks stay on one shared stream of draws.
extern "C" __global__ __aicore__ void sample_seed_bump_kernel(GM_ADDR S){
 if(GetBlockIdx()!=0)return;
 GlobalTensor<int32_t> sg;sg.SetGlobalBuffer((__gm__ int32_t*)S);
 TPipe pipe;TQue<QuePosition::VECOUT,1> qo;pipe.InitBuffer(qo,1,32);
 TBuf<TPosition::VECCALC> bs;pipe.InitBuffer(bs,32);auto sv=bs.Get<int32_t>();
 DataCopyPad(sv,sg,DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 int32_t next=sv.GetValue(0)+1;
 auto out=qo.AllocTensor<int32_t>();out.SetValue(0,next);
 qo.EnQue(out);out=qo.DeQue<int32_t>();
 DataCopyPad(sg,out,DataCopyExtParams{1,4,0,0,0});qo.FreeTensor(out);
}

// Single writer packs the compact wire format, including odd-B 48-byte tails.
extern "C" __global__ __aicore__ void greedy_local_join_kernel(
    GM_ADDR P,GM_ADDR O,uint32_t rows){
 if(GetBlockIdx()!=0)return;
 GlobalTensor<float> p,o;p.SetGlobalBuffer((__gm__ float*)P);o.SetGlobalBuffer((__gm__ float*)O);
 TPipe pipe;TBuf<TPosition::VECCALC> bp;
 pipe.InitBuffer(bp,GSPLIT*GLANES*4);auto v=bp.Get<float>();
 TQue<QuePosition::VECOUT,1> qo;pipe.InitBuffer(qo,1,(rows*2*4+31)/32*32);
 auto out=qo.AllocTensor<float>();
 for(uint32_t row=0;row<rows;row++){
  DataCopy(v,p[uint64_t(row)*GSPLIT*GLANES],int32_t(GSPLIT*GLANES));
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  float best=v.GetValue(0),id=v.GetValue(1);bool bad=id<0;
  for(uint32_t j=1;j<GSPLIT;j++){
   float f=v.GetValue(j*GLANES),i=v.GetValue(j*GLANES+1);bad=bad||i<0;
   if(f>best||(f==best&&i<id)){best=f;id=i;}
  }
  out.SetValue(row*2,bad?0.0f:best);out.SetValue(row*2+1,bad?-1.0f:id);
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 }
 qo.EnQue(out);out=qo.DeQue<float>();
 DataCopyPad(o,out,DataCopyExtParams{1,rows*2*4,0,0,0});qo.FreeTensor(out);
}

extern "C" __global__ __aicore__ void greedy_global_kernel(
    GM_ADDR P,GM_ADDR O,uint32_t rows,uint32_t row_stride,uint32_t row_offset){
 if(GetBlockIdx()!=0)return;
 GlobalTensor<float> p;p.SetGlobalBuffer((__gm__ float*)P);
 GlobalTensor<int64_t> o;o.SetGlobalBuffer((__gm__ int64_t*)O);
 TPipe pipe;TBuf<TPosition::VECCALC> bp;
 pipe.InitBuffer(bp,GWORLD*rows*2*4);auto v=bp.Get<float>();
 TQue<QuePosition::VECOUT,1> qo;pipe.InitBuffer(qo,1,(rows*row_stride*8+31)/32*32);
 DataCopy(v,p,int32_t(GWORLD*rows*2));
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 auto out=qo.AllocTensor<int64_t>();
 for(uint32_t i=0;i<rows*row_stride;++i)out.SetValue(i,-1);
 for(uint32_t row=0;row<rows;row++){
  float best=0.0f;int64_t id=-1;bool bad=false;
  for(uint32_t rank=0;rank<GWORLD;rank++){
   uint32_t k=(rank*rows+row)*2;
   float f=v.GetValue(k),fi=v.GetValue(k+1);
   // Test bounds before casting (NaN/inf/malformed FP32 ids are never cast).
   if(f!=f||!(fi>=float(int32_t(rank*GV))&&fi<float(int32_t((rank+1)*GV)))){bad=true;continue;}
   int64_t i=int64_t(fi);if(float(i)!=fi){bad=true;continue;}
   if(id<0||f>best||(f==best&&i<id)){best=f;id=i;}
  }
  out.SetValue(row*row_stride+row_offset,bad?-1:id);
 }
 qo.EnQue(out);out=qo.DeQue<int64_t>();
 DataCopyPad(o,out,DataCopyExtParams{1,rows*row_stride*8,0,0,0});qo.FreeTensor(out);
}

// Pure host validation: only caller-provided addresses/sizes, never device reads.
static int greedy_ranges(void* const* p,const uint64_t* n,uint32_t count){
 for(uint32_t i=0;i<count;i++){
  uintptr_t a=uintptr_t(p[i]);
  if(!a||a%32||a>UINTPTR_MAX-n[i])return -1;
  for(uint32_t j=0;j<i;j++){
   uintptr_t b=uintptr_t(p[j]);if(a<b+n[j]&&b<a+n[i])return -2;
  }
 }
 return 0;
}
extern "C" uint64_t dec_greedy_workspace_bytes(uint32_t batch){
 return !batch||batch>64?0:uint64_t(batch)*Q*GSPLIT*GLANES*4;
}
extern "C" int dec_greedy_local(void* stream,void* logits,void* partial,
    void* pair,uint32_t batch,uint32_t rank){
 if(!stream||!batch||batch>64||rank>=GWORLD)return -1;
 uint32_t rows=batch*Q;void* p[]={logits,partial,pair};
 uint64_t n[]={uint64_t(rows)*GV*4,uint64_t(rows)*GSPLIT*GLANES*4,uint64_t(rows)*2*4};
 int rc=greedy_ranges(p,n,3);if(rc)return rc;
 uint32_t blocks=rows*GSPLIT<40?rows*GSPLIT:40;
 greedy_part_kernel<<<blocks,nullptr,stream>>>((uint8_t*)logits,(uint8_t*)partial,rows,rank,blocks,1,0);
 greedy_local_join_kernel<<<1,nullptr,stream>>>((uint8_t*)partial,(uint8_t*)pair,rows);
 return 0;
}
extern "C" int dec_greedy_global(void* stream,void* gathered,void* greedy,uint32_t batch){
 if(!stream||!batch||batch>64)return -1;
 void* p[]={gathered,greedy};uint64_t n[]={uint64_t(batch)*Q*GWORLD*2*4,uint64_t(batch)*Q*8};
 int rc=greedy_ranges(p,n,2);if(rc)return rc;
 greedy_global_kernel<<<1,nullptr,stream>>>((uint8_t*)gathered,(uint8_t*)greedy,batch*Q,1,0);
 return 0;
}
extern "C" int dec_greedy_accept(void* stream,void* gathered,void* ids,void* meta,
    void* greedy,void* result,uint32_t batch,uint32_t slots){
 if(!stream||!batch||batch>64||!slots)return -1;
 void* p[]={gathered,ids,meta,greedy,result};
 uint64_t n[]={uint64_t(batch)*Q*GWORLD*2*4,uint64_t(batch)*Q*8,
               uint64_t(batch)*M*8,uint64_t(batch)*Q*8,uint64_t(batch)*M*8};
 int rc=greedy_ranges(p,n,5);if(rc)return rc;
 rc=dec_greedy_global(stream,gathered,greedy,batch);if(rc)return rc;
 return dec_accept(stream,ids,greedy,meta,result,batch,slots);
}
// Exact ABI of CANN HcclAllGather: send,recv,count,HcclDataType,comm,stream.
// Function address is supplied explicitly so build.sh needs no new link flags.
using DecodeAllGather=int (*)(void*,void*,uint64_t,int,void*,void*);
extern "C" int dec_greedy_tp8(void* stream,void* logits,void* partial,void* pair,
    void* gathered,void* greedy,void* comm,DecodeAllGather all_gather,
    uint32_t batch,uint32_t rank,uint32_t world){
 if(!stream||!comm||!all_gather||!batch||batch>64||rank>=GWORLD||world!=GWORLD)return -1;
 uint64_t rows=uint64_t(batch)*Q;void* p[]={logits,partial,pair,gathered,greedy};
 uint64_t n[]={rows*GV*4,rows*GSPLIT*GLANES*4,rows*2*4,rows*GWORLD*2*4,rows*8};
 int rc=greedy_ranges(p,n,5);if(rc)return rc;
 rc=dec_greedy_local(stream,logits,partial,pair,batch,rank);if(rc)return rc;
 rc=all_gather(pair,gathered,rows*2,4,comm,stream);if(rc)return rc;
 return dec_greedy_global(stream,gathered,greedy,batch);
}

// Sampling variants. Extra caller-owned buffers, same disjointness rule:
// temp FP32 [B]   per-request temperature, 0 (or NaN/absurd) means greedy
// seed INT32 [1]  shared draw counter, identical on every rank, bumped here
// A temp of all zeros reproduces dec_greedy_* token for token.
extern "C" int dec_sample_local(void* stream,void* logits,void* partial,void* pair,
    void* temp,void* seed,uint32_t batch,uint32_t rank){
 if(!stream||!batch||batch>64||rank>=GWORLD)return -1;
 uint32_t rows=batch*Q;void* p[]={logits,partial,pair,temp,seed};
 uint64_t n[]={uint64_t(rows)*GV*4,uint64_t(rows)*GSPLIT*GLANES*4,uint64_t(rows)*2*4,
               uint64_t(batch)*4,4};
 int rc=greedy_ranges(p,n,5);if(rc)return rc;
 uint32_t blocks=rows*GSPLIT<40?rows*GSPLIT:40;
 sample_part_kernel<<<blocks,nullptr,stream>>>((uint8_t*)logits,(uint8_t*)partial,
     (uint8_t*)temp,(uint8_t*)seed,rows,rank,blocks,Q);
 greedy_local_join_kernel<<<1,nullptr,stream>>>((uint8_t*)partial,(uint8_t*)pair,rows);
 return 0;
}
extern "C" int dec_sample_tp8(void* stream,void* logits,void* partial,void* pair,
    void* gathered,void* greedy,void* temp,void* seed,void* comm,
    DecodeAllGather all_gather,uint32_t batch,uint32_t rank,uint32_t world){
 if(!stream||!comm||!all_gather||!batch||batch>64||rank>=GWORLD||world!=GWORLD)return -1;
 uint64_t rows=uint64_t(batch)*Q;
 void* p[]={logits,partial,pair,gathered,greedy,temp,seed};
 uint64_t n[]={rows*GV*4,rows*GSPLIT*GLANES*4,rows*2*4,rows*GWORLD*2*4,rows*8,
               uint64_t(batch)*4,4};
 int rc=greedy_ranges(p,n,7);if(rc)return rc;
 rc=dec_sample_local(stream,logits,partial,pair,temp,seed,batch,rank);if(rc)return rc;
 rc=all_gather(pair,gathered,rows*2,4,comm,stream);if(rc)return rc;
 rc=dec_greedy_global(stream,gathered,greedy,batch);if(rc)return rc;
 // After every reader of this step, so the next step draws a fresh stream.
 sample_seed_bump_kernel<<<1,nullptr,stream>>>((uint8_t*)seed);
 return 0;
}

// Accepted-prefix transaction. Static descriptors are built/validated by Commit
// before capture, kept immutable and alive with all pointees (no pointer arena).
// desc[N,12] int64: kind(0=window,1/2=source), pending, bank,
// index_pending,index_bank,values,gates,carry_values,carry_scores,0,0,0.
// checked[B,8] copies result with a batch-wide error in col5. Any error prevents
// ALL writes, including pos. Codes: 101 metadata,102 stale pos,103 duplicate slot,
// 104 missing/out-of-range page,105 upstream error,106 aliased page ownership.
// Dynamic inputs/page table must not change during this ordered 3-kernel call.
namespace PrefixCommit {
__aicore__ inline void read_meta(LocalTensor<int64_t> dst, GM_ADDR src, uint32_t n){
 GlobalTensor<int64_t> g;g.SetGlobalBuffer((__gm__ int64_t*)src);
 DataCopy(dst,g,int32_t(n));
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}
__aicore__ inline int64_t page_at(GlobalTensor<int64_t>& table,LocalTensor<int64_t> tmp,uint64_t i){
 DataCopyPad(tmp,table[i],DataCopyExtParams{1,8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 int64_t v=tmp.GetValue(0);
 SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 return v;
}
__global__ __aicore__ void validate(GM_ADDR R,GM_ADDR P,GM_ADDR T,GM_ADDR C,
 uint32_t B,uint32_t S,uint32_t pages,uint32_t mp,uint32_t pt,uint32_t limit,uint32_t sources){
 if(GetBlockIdx()!=0)return;
 TPipe pipe;TBuf<TPosition::VECCALC> mb,pb,tb;
 pipe.InitBuffer(mb,4*64);pipe.InitBuffer(pb,32);pipe.InitBuffer(tb,256*8);
 auto m=mb.Get<int64_t>();auto pos=pb.Get<int32_t>();auto tmp=tb.Get<int64_t>();
 read_meta(m,R,B*M);
 GlobalTensor<int32_t> pg;pg.SetGlobalBuffer((__gm__ int32_t*)P);
 GlobalTensor<int64_t> table,out;table.SetGlobalBuffer((__gm__ int64_t*)T);out.SetGlobalBuffer((__gm__ int64_t*)C);
 int64_t err=0,ids[8],owners[8];uint32_t used=0;
 for(uint32_t b=0;b<B&&!err;b++){
  uint32_t off=b*M;int64_t active=m.GetValue(off+4),slot=m.GetValue(off),start=m.GetValue(off+1),n=m.GetValue(off+2);
  if(m.GetValue(off+5)){err=105;break;}
  if(active!=0&&active!=1){err=101;break;}if(!active)continue;
  if(slot<0||slot>=S||start<0||start>limit||n<0||n>Q||n>int64_t(limit)-start||
     (n>0&&(m.GetValue(off+3)<0||m.GetValue(off+3)>=129280))){err=101;break;}
  for(uint32_t j=0;j<b;j++)if(m.GetValue(j*M+4)&&m.GetValue(j*M)==slot)err=103;
  if(err)break;
  DataCopyPad(pos,pg[slot],DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  if(pos.GetValue(0)!=start){err=102;break;}
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
  if(!sources||!n)continue;
  // Even page_tokens >= Q: at most two token pages per accepted row.
  for(int64_t page=start/pt;page<=(start+n-1)/pt;page++){
   if(page>=mp){err=104;break;}
   int64_t owner=slot*mp+page,id=page_at(table,tmp,owner);
   if(id<0||id>=pages){err=104;break;}
   ids[used]=id;owners[used++]=owner;
  }
 }
 // Enforce exclusive physical page ownership, including inactive slots and old
 // logical pages: a malformed table must not corrupt another request's history.
 for(uint64_t off=0;off<uint64_t(S)*mp&&!err&&used;off+=256){
  uint32_t n=uint64_t(S)*mp-off<256?uint64_t(S)*mp-off:256;
  DataCopyPad(tmp,table[off],DataCopyExtParams{1,n*8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  for(uint32_t j=0;j<n&&!err;j++)for(uint32_t k=0;k<used;k++)
   if(tmp.GetValue(j)==ids[k]&&int64_t(off+j)!=owners[k]){err=106;break;}
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 }
 for(uint32_t b=0;b<B;b++)m.SetValue(b*M+5,err);
 SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
 DataCopy(out,m,int32_t(B*M));
}
template<class V> __aicore__ inline void copy_row(LocalTensor<V> tmp,int64_t src,uint64_t si,int64_t dst,uint64_t di,uint32_t dim){
 GlobalTensor<V> s,d;s.SetGlobalBuffer((__gm__ V*)src);d.SetGlobalBuffer((__gm__ V*)dst);
 DataCopy(tmp,s[si*dim],int32_t(dim));
 SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
 DataCopy(d[di*dim],tmp,int32_t(dim));PipeBarrier<PIPE_ALL>();
}
__global__ __aicore__ void apply(GM_ADDR C,GM_ADDR D,GM_ADDR T,uint32_t B,uint32_t N,
 uint32_t mp,uint32_t pt,uint32_t ring,uint32_t pad,uint32_t blocks){
 if(GetBlockIdx()>=blocks)return;
 TPipe pipe;TBuf<TPosition::VECCALC> mb,db,tb,rb;
 pipe.InitBuffer(mb,4*64);pipe.InitBuffer(db,96);pipe.InitBuffer(tb,32);pipe.InitBuffer(rb,2048);
 auto m=mb.Get<int64_t>(),d=db.Get<int64_t>(),tmp=tb.Get<int64_t>();
 read_meta(m,C,B*M);if(m.GetValue(5))return;
 GlobalTensor<int64_t> desc,table;desc.SetGlobalBuffer((__gm__ int64_t*)D);table.SetGlobalBuffer((__gm__ int64_t*)T);
 for(uint32_t task=GetBlockIdx();task<N*B;task+=blocks){
  uint32_t b=task%B;int64_t slot=m.GetValue(b*M),start=m.GetValue(b*M+1),n=m.GetValue(b*M+2);
  if(!m.GetValue(b*M+4)||!n)continue;
  DataCopy(d,desc[(task/B)*12],12);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  int64_t kind=d.GetValue(0),p=d.GetValue(1),bank=d.GetValue(2);
  if(kind==0){
   for(int64_t q=0;q<n;q++)copy_row(rb.Get<uint16_t>(),p,b*Q+q,bank,
       uint64_t(slot)*(uint64_t(pad)+ring)+pad+(start+q)%ring,512);
  }else{
   int64_t first=start/kind,count=(start+n)/kind-first,rpp=pt/kind;
   for(int64_t q=0;q<count;q++){
    int64_t row=first+q,page=page_at(table,tmp,slot*mp+row/rpp);
    uint64_t dst=uint64_t(page)*rpp+row%rpp;
    copy_row(rb.Get<uint16_t>(),p,b*Q+q,bank,dst,512);
    copy_row(rb.Get<uint16_t>(),d.GetValue(3),b*Q+q,d.GetValue(4),dst,128);
   }
   if(kind==2)for(int64_t q=n>4?n-4:0;q<n;q++){
    uint64_t dst=uint64_t(slot)*4+(start+q)%4;
    copy_row(rb.Get<uint32_t>(),d.GetValue(5),b*Q+q,d.GetValue(7),dst,512);
    copy_row(rb.Get<uint32_t>(),d.GetValue(6),b*Q+q,d.GetValue(8),dst,512);
   }
  }
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
 }
}
__global__ __aicore__ void publish(GM_ADDR C,GM_ADDR P,uint32_t B){
 if(GetBlockIdx()!=0)return;
 TPipe pipe;TBuf<TPosition::VECCALC> mb,pb;pipe.InitBuffer(mb,4*64);pipe.InitBuffer(pb,32);
 auto m=mb.Get<int64_t>();auto pos=pb.Get<int32_t>();read_meta(m,C,B*M);
 if(m.GetValue(5))return;
 GlobalTensor<int32_t> pg;pg.SetGlobalBuffer((__gm__ int32_t*)P);
 for(uint32_t b=0;b<B;b++)if(m.GetValue(b*M+4)&&m.GetValue(b*M+2)){
  pos.SetValue(0,int32_t(m.GetValue(b*M+1)+m.GetValue(b*M+2)));
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopyPad(pg[m.GetValue(b*M)],pos,DataCopyExtParams{1,4,0,0,0});
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
 }
}
} // namespace PrefixCommit
extern "C" int dec_commit_prefix(void* stream,void* result,void* pos,void* table,
 void* descriptors,void* checked,uint32_t batch,uint32_t slots,uint32_t pages,
 uint32_t maxpages,uint32_t page_tokens,uint32_t ring,uint32_t pad,uint32_t max_seq,
 uint32_t windows,uint32_t sources){
 if(!stream||!batch||batch>4||slots<batch||!pages||!maxpages||page_tokens<Q||page_tokens%2||
    ring<144||pad<16||uint64_t(ring)+pad>UINT32_MAX||!max_seq||max_seq>INT32_MAX||
    maxpages!=(uint64_t(max_seq)+page_tokens-1)/page_tokens||windows>43||sources>4||!windows)return -1;
 void* p[]={result,pos,table,descriptors,checked};
 uint64_t n[]={uint64_t(batch)*64,uint64_t(slots)*4,uint64_t(slots)*maxpages*8,
               uint64_t(windows+sources)*96,uint64_t(batch)*64};
 int rc=greedy_ranges(p,n,5);if(rc)return rc;
 PrefixCommit::validate<<<1,nullptr,stream>>>((uint8_t*)result,(uint8_t*)pos,(uint8_t*)table,
     (uint8_t*)checked,batch,slots,pages,maxpages,page_tokens,max_seq,sources);
 uint32_t blocks=(windows+sources)*batch<40?(windows+sources)*batch:40;
 PrefixCommit::apply<<<blocks,nullptr,stream>>>((uint8_t*)checked,(uint8_t*)descriptors,(uint8_t*)table,
     batch,windows+sources,maxpages,page_tokens,ring,pad,blocks);
 PrefixCommit::publish<<<1,nullptr,stream>>>((uint8_t*)checked,(uint8_t*)pos,batch);
 return 0;
}

// Draft chain uses only row b*6+step. Compact partial/pair outputs.
extern "C" int dec_draft_greedy_local(void* stream,void* logits,void* partial,
    void* pair,uint32_t batch,uint32_t rank,uint32_t step){
 if(!stream||!batch||batch>4||rank>=GWORLD||step>=5)return -1;
 void* p[]={logits,partial,pair};
 uint64_t n[]={uint64_t(batch)*Q*GV*4,uint64_t(batch)*GSPLIT*GLANES*4,uint64_t(batch)*2*4};
 int rc=greedy_ranges(p,n,3);if(rc)return rc;
 uint32_t blocks=batch*GSPLIT<40?batch*GSPLIT:40;
 greedy_part_kernel<<<blocks,nullptr,stream>>>((uint8_t*)logits,(uint8_t*)partial,batch,rank,blocks,6,step);
 greedy_local_join_kernel<<<1,nullptr,stream>>>((uint8_t*)partial,(uint8_t*)pair,batch);
 return 0;
}

// Compact draft collective, preserving padded [B,6] chain output ABI.
extern "C" int dec_draft_greedy_tp8(void* stream,void* logits,void* partial,void* pair,
    void* gathered,void* greedy,void* comm,DecodeAllGather all_gather,
    uint32_t batch,uint32_t rank,uint32_t world,uint32_t step){
 if(!stream||!comm||!all_gather||!batch||batch>4||rank>=GWORLD||world!=GWORLD||step>=5)return -1;
 uint64_t rows=batch;void* p[]={logits,partial,pair,gathered,greedy};
 uint64_t n[]={rows*Q*GV*4,rows*GSPLIT*GLANES*4,rows*2*4,rows*GWORLD*2*4,rows*Q*8};
 int rc=greedy_ranges(p,n,5);if(rc)return rc;
 rc=dec_draft_greedy_local(stream,logits,partial,pair,batch,rank,step);if(rc)return rc;
 rc=all_gather(pair,gathered,rows*2,4,comm,stream);if(rc)return rc;
 greedy_global_kernel<<<1,nullptr,stream>>>((uint8_t*)gathered,(uint8_t*)greedy,batch,6,step);
 return 0;
}
