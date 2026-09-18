cd /mnt/data/kw/ljqinfer_dsv4f_tp8
rm -f ops/.build/lock
echo "=== GATE"; BMAX=2 NSTEP=16 /mnt/data/kw/anaconda3/bin/torchrun --nproc_per_node=8 bench/tc_step_batch.py 2>&1 | grep -v Warning | tail -15
echo "=== PERF"; BLIST=1,2,4 /mnt/data/kw/anaconda3/bin/torchrun --nproc_per_node=8 bench/tc_amdahl_b.py 2>&1 | grep -v Warning | tail -25
echo A7_DONE
