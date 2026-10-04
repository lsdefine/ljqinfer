// Host-only tiling generator for the paged indexer scoring GEMM.
// A is the packed indexer queries (M = 6 MTP rows x 4 heads = 24, K = 128 bf16).
// B is one committed KV page of the indexer bank, transposed (N = rows per page,
// K = 128 bf16); a page is fully contiguous in global memory so a single
// IterateAll consumes it. C is FP32 (M, N) written into a per-page block.
// Rows per page is PAGE_TOKENS / ratio, so only 1024 and 2048 can occur.
#include "matmul_tiling.h"
#include <fstream>
#include <iostream>
#include <vector>
int main() {
    using namespace matmul_tiling;
    for (int n : {1024, 2048}) {
        MatmulApiTiling api;
        api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
        api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, true);
        api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        api.SetShape(24, n, 128);
        api.SetOrgShape(24, n, 128);
        api.SetBias(false);
        api.SetFixSplit(32, 256, 64);
        // Conservative planning limits, not measured free device memory.
        api.SetBufferSpace(256*1024, 64*1024, 128*1024);
        optiling::TCubeTiling t;
        auto rc = api.GetTiling(t);
        if (rc != 0) { std::cerr << "GetTiling failed N=" << n << " rc=" << rc << "\n"; return 1; }
        std::cout << "N=" << n << " singleCore=" << t.get_singleCoreM() << "," << t.get_singleCoreN() << "," << t.get_singleCoreK()
                  << " base=" << t.get_baseM() << "," << t.get_baseN() << "," << t.get_baseK()
                  << " org=" << t.get_M() << "," << t.get_N() << "," << t.get_Ka() << "," << t.get_Kb()
                  << " bytes=" << t.GetDataSize() << std::endl;
        std::vector<unsigned char> data(t.GetDataSize());
        t.SaveToBuffer(data.data(), data.size());
        std::ofstream f("source_cube_tiling_n" + std::to_string(n) + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
        if (!f) return 2;
    }
}
