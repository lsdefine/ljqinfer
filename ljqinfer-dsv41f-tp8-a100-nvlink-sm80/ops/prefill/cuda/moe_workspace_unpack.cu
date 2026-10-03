#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
struct Sources { const unsigned char* w[48]; const unsigned char* s[48]; };
__global__ void direct_kernel(Sources src,const unsigned short* __restrict__ lut,unsigned int* __restrict__ out,int64_t n,int64_t ebase){
 int e=blockIdx.y;
 for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x){
  unsigned int v=src.w[e][i],scale=src.s[e][i/16];
  out[(ebase+e)*n+i]=(unsigned int)lut[scale*16+(v&15)]|((unsigned int)lut[scale*16+(v>>4)]<<16);
 }
}
torch::Tensor unpack(std::vector<torch::Tensor> w,std::vector<torch::Tensor> s,torch::Tensor lut,torch::Tensor y){
 TORCH_CHECK(w.size()>0 && w.size()==s.size(),"matching expert lists required");
 TORCH_CHECK(lut.is_cuda() && lut.is_contiguous() && lut.scalar_type()==torch::kBFloat16 && lut.numel()==4096,"BF16 LUT required");
 c10::cuda::CUDAGuard guard(lut.device());
 int64_t n=w[0].numel(),k=w[0].size(-1)*2;
 TORCH_CHECK(n>0 && k%32==0,"positive aligned shape required");
 std::vector<Sources> batches((w.size()+47)/48);
 for(size_t e=0;e<w.size();++e){
  Sources& src=batches[e/48];
  size_t slot=e%48;
  TORCH_CHECK(w[e].device()==lut.device() && s[e].device()==lut.device(),"same device required");
  TORCH_CHECK(w[e].is_contiguous() && s[e].is_contiguous(),"contiguous expert required");
  TORCH_CHECK(w[e].scalar_type()==torch::kInt8 && s[e].scalar_type()==torch::kUInt8,"canonical FP4/E8M0 required");
  TORCH_CHECK(w[e].sizes()==w[0].sizes() && s[e].numel()*16==n,"expert geometry mismatch");
  src.w[slot]=(const unsigned char*)w[e].data_ptr();src.s[slot]=s[e].data_ptr<unsigned char>();
 }
 TORCH_CHECK(y.device()==lut.device() && y.scalar_type()==torch::kBFloat16 && y.is_contiguous(),"contiguous rank-local BF16 output required");
 TORCH_CHECK(y.dim()==3 && y.size(0)==(int64_t)w.size() && y.size(1)==n*2/k && y.size(2)==k,"output geometry mismatch");
 TORCH_CHECK(!y.is_alias_of(lut),"output must not alias LUT");
 for(size_t e=0;e<w.size();++e){
  TORCH_CHECK(!y.is_alias_of(w[e]) && !y.is_alias_of(s[e]),"output must not alias packed inputs");
 }
 for(size_t b=0;b<batches.size();++b){
  int64_t ebase=(int64_t)b*48, rows=std::min<int64_t>(48,(int64_t)w.size()-ebase);
  dim3 grid(std::min<int64_t>((n+255)/256,1024),(unsigned)rows);
  direct_kernel<<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(
    batches[b],(const unsigned short*)lut.data_ptr(),(unsigned int*)y.data_ptr(),n,ebase);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("unpack",&unpack);}
