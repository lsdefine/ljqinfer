#include <cstdint>
#include <arm_neon.h>
static inline uint16x4_t cvt(int32x4_t v,float32x4_t scale) {
 uint32x4_t bits=vreinterpretq_u32_f32(vmulq_f32(vcvtq_f32_s32(v),scale));
 uint32x4_t nan=vcgtq_u32(vandq_u32(bits,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
 uint32x4_t rounded=vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fff),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1))));
 return vmovn_u32(vbslq_u32(nan,vdupq_n_u32(0x7fc0),vshrq_n_u32(rounded,16)));
}
extern "C" int gather_bf16(const int8_t* w,const float* scales,const int64_t* ids,uint16_t* out,int64_t count,int64_t rows) {
 for(int64_t i=0;i<count;i++) if(ids[i]<0 || ids[i]>=rows) return 1;
 #pragma omp parallel for num_threads(4) schedule(static) if(count>=3072)
 for(int64_t i=0;i<count;i++) {
  if(i+8<count) {
   const int8_t* next=w+ids[i+8]*256;
   for(int k=0;k<256;k+=64) __builtin_prefetch(next+k,0,1);
   __builtin_prefetch(scales+ids[i+8]*8,0,1);
  }
  const int64_t row=ids[i];
  for(int g=0;g<8;g++) {
   float32x4_t scale=vdupq_n_f32(scales[row*8+g]);
   for(int j=0;j<32;j+=16) {
    int8x16_t v=vld1q_s8(w+row*256+g*32+j);
    int16x8_t lo=vmovl_s8(vget_low_s8(v)),hi=vmovl_s8(vget_high_s8(v));
    uint16_t* dst=out+i*256+g*32+j;
    vst1_u16(dst,cvt(vmovl_s16(vget_low_s16(lo)),scale));
    vst1_u16(dst+4,cvt(vmovl_s16(vget_high_s16(lo)),scale));
    vst1_u16(dst+8,cvt(vmovl_s16(vget_low_s16(hi)),scale));
    vst1_u16(dst+12,cvt(vmovl_s16(vget_high_s16(hi)),scale));
   }
  }
 }
 return 0;
}
