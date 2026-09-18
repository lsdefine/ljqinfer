# -*- coding: utf-8 -*-
"""Build oracle weight tree via load_tp8 (clean path) and snapshot as final-tree.
nohup python snapshot_oracle_tree.py > /tmp/snap_tree.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import time, traceback
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

try:
    from model.weights import load_tp8
    from model import wcache
    log('building tree via load_tp8 (clean oracle path)...')
    tree = load_tp8()
    log('tree built; snapshotting...')
    m = wcache.snapshot_tree(tree)
    log('snapshot done; manifest keys:', {k: v for k, v in m.items() if not isinstance(v, (dict, list))})
    log('SNAP OK')
except Exception:
    traceback.print_exc()
    log('SNAP FAILED')
