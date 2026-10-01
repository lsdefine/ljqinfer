import torch,time,json,statistics
from model.sampling import sample_pairs
result=[]
for rows in (1,8,32):
 x=torch.randn(rows,151936//4,device='cuda',dtype=torch.float32)
 def old():
  v,i=x.max(-1);return torch.stack((v,i.float()),-1)
 def new():return sample_pairs(x,[1.],0,4)
 for _ in range(20):old();new()
 measurements={}
 for name,fn in [('old',old),('t1',new)]:
  wall=[];dev=[]
  for _ in range(5):
   torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   t=time.perf_counter();a.record()
   for k in range(100):fn()
   b.record();b.synchronize();wall.append((time.perf_counter()-t)*1e6/100);dev.append(a.elapsed_time(b)*1000/100)
  measurements[name]={'wall_us':statistics.median(wall),'event_us':statistics.median(dev)}
 result.append({'rows':rows,**measurements})
print(json.dumps(result))
