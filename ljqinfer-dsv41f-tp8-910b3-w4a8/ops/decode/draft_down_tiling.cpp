// Host-only tiling generator: no ACL initialization and no device execution.
#include "matmul_tiling.h"
#include <fstream>
#include <iostream>
#include <vector>
int main() {
    using namespace matmul_tiling;
    for (int m : {6,12,18,24}) {
        MatmulApiTiling api;
        api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
        api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, true);
        api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        api.SetShape(m,256,288);
        api.SetOrgShape(m,5120,288);
        api.SetBias(false);
        api.SetFixSplit(m<=16 ? 16 : 32,256,64);
        // Conservative planning limits, not measured free device memory.
        api.SetBufferSpace(256*1024,64*1024,128*1024);
        optiling::TCubeTiling t;
        auto rc=api.GetTiling(t);
        if(rc!=0) { std::cerr<<"GetTiling failed M="<<m<<" rc="<<rc<<"\n"; return 1; }
        std::cout<<"M="<<m<<" singleCore="<<t.get_singleCoreM()<<","<<t.get_singleCoreN()<<","<<t.get_singleCoreK()
                 <<" base="<<t.get_baseM()<<","<<t.get_baseN()<<","<<t.get_baseK()
                 <<" org="<<t.get_M()<<","<<t.get_N()<<","<<t.get_Ka()<<","<<t.get_Kb()
                 <<" bytes="<<t.GetDataSize()<<std::endl;
        std::vector<unsigned char> data(t.GetDataSize());
        t.SaveToBuffer(data.data(),data.size());
        std::ofstream f("draft_down_tiling_m"+std::to_string(m)+".bin",std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),data.size());
        if(!f) return 2;
    }
}
