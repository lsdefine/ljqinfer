import io,sys
src=io.open('ops/decode/cuda/moe_decode_fp4.cu',encoding='utf-8').read()
A_old="""  extern __shared__ __nv_bfloat16 ms[];
  for (int i = threadIdx.x; i < TOPK * M; i += blockDim.x)
    ms[i] = mid[(size_t)t * TOPK * M + i];
  __syncthreads();"""
A_new="""  extern __shared__ __nv_bfloat16 ms[];
  // Pad each 32-value block to 40 bf16 (80 B).  The 8 distinct shm addresses a
  // warp touches then map onto all 32 banks (20*L % 32) instead of only two
  // groups (16*L % 32 -> 4-way conflict), and 80 B stays 16 B aligned so the
  // four ms2[jj] reads still fuse into one LDS.128.  Layout only: the order of
  // every float op is untouched, so the result is bit-identical.
  const int MSB = 40;
  const int MSK = (M >> 5) * MSB;
  for (int i = threadIdx.x; i < TOPK * M; i += blockDim.x) {
    const int kt = i / M, r = i - kt * M;
    ms[kt * MSK + (r >> 5) * MSB + (r & 31)] = mid[(size_t)t * TOPK * M + i];
  }
  __syncthreads();"""
B_old="        const __nv_bfloat162* ms2 = (const __nv_bfloat162*)(ms + ktop * M + mb);"
B_new="        const __nv_bfloat162* ms2 =\n            (const __nv_bfloat162*)(ms + ktop * MSK + q * MSB + (j << 3));"
C_old="  const size_t dshm = (size_t)(topk * Nff) * sizeof(__nv_bfloat16);"
C_new="  const size_t dshm = (size_t)(topk * ((Nff >> 5) * 40)) * sizeof(__nv_bfloat16);"
for o,n in ((A_old,A_new),(B_old,B_new),(C_old,C_new)):
    assert src.count(o)==1, ('MISS/DUP %d'%src.count(o), o[:60])
    src=src.replace(o,n)
io.open('tests/moe_pad.cu','w',encoding='utf-8').write(src)
print('tests/moe_pad.cu written', len(src))
