#include <torch/extension.h>
torch::Tensor ptr_bgemm_cuda(torch::Tensor x,torch::Tensor ids,torch::Tensor ew,torch::Tensor wg,torch::Tensor wu,torch::Tensor wd);
torch::Tensor decode(torch::Tensor x,torch::Tensor ids,torch::Tensor ew,torch::Tensor wg,torch::Tensor wu,torch::Tensor wd){return ptr_bgemm_cuda(x,ids,ew,wg,wu,wd);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode",&decode);}
