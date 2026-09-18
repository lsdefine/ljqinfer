
from tokenizers import Tokenizer
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
exec(open('/mnt/data/kw/ljqinfer_dsv4f_tp8/_needle_probe.py').read().split("def chat")[0])
tk=Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
doc=filler(375,1)
ids=tk.encode(doc).ids
print(len(doc.split()),'words ->',len(ids),'tokens; first 20:',[tk.decode([i]) for i in ids[:20]])
