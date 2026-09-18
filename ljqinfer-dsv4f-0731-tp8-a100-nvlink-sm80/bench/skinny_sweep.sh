cd /mnt/data/kw/ljqinfer_dsv4f_tp8
for cfg in "8 1" "8 2" "8 4" "16 2"; do
  set -- $cfg
  sed -i "s/^#define SKINNY_MT .*/#define SKINNY_MT $1/; s/^#define SKINNY_NT .*/#define SKINNY_NT $2/" ops/dsv4_wgemm.cu
  rm -f ops/.build/lock
  echo "=== v2 MT=$1 NT=$2"
  CUDA_VISIBLE_DEVICES=0 python3 -u bench/skinny_bench.py cmp 2>&1 | grep -E "7168x1024x(1|8|16|32) |7168x512x(8|32) |2048x512x(8|32) |rror"
done
echo SWEEP_DONE
