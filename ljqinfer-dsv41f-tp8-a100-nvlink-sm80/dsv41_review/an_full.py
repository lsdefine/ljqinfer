import json
rows=[l for l in open("/tmp/serve_384k/worker.log",errors="ignore") if "PROFJSON" in l]
d=json.loads(rows[0][rows[0].index("{"):].strip()); S=10.0
o=sorted(((float(v[0] if isinstance(v,list) else v),n,(v[1] if isinstance(v,list) and len(v)>1 else 0)) for n,v in d.items()),reverse=True)
print("N_KERNEL_TYPES",len(o),"TOTAL %.3f ms  LAUNCH/step %.0f"%(sum(x[0] for x in o)/S,sum(x[2] for x in o)/S))
for i,(ms,n,c) in enumerate(o):
    if i>=24: print("%8.4f %6.1f  %s"%(ms/S,c/S,n[:62]))
print("TAIL_SUM %.3f over %d kernels"%(sum(x[0] for x in o[24:])/S,len(o)-24))
