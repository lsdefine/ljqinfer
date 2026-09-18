cd /mnt/data/kw/ljqinfer_dsv4f_tp8
cp ops/attn_decode_g.py /tmp/a8_attn_decode_g.py; cp ops/__init__.py /tmp/a8_init.py
git checkout ops/attn_decode_g.py ops/__init__.py
echo "=== OLD"; BLIST=1,2,4 /mnt/data/kw/anaconda3/bin/torchrun --nproc_per_node=8 bench/tc_amdahl_b.py 2>&1 | grep -v warn | grep 'graph '
cp /tmp/amdahl_b.json /tmp/amdahl_b_a8_old.json
cp /tmp/a8_attn_decode_g.py ops/attn_decode_g.py; cp /tmp/a8_init.py ops/__init__.py
echo "=== NEW"; BLIST=1,2,4 /mnt/data/kw/anaconda3/bin/torchrun --nproc_per_node=8 bench/tc_amdahl_b.py 2>&1 | grep -v warn | grep 'graph '
cp /tmp/amdahl_b.json /tmp/amdahl_b_a8_new.json
echo AB_DONE
