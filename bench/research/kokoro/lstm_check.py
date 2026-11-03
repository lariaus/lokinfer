import time, statistics, torch
nn=torch.nn
for dev in ('cpu','mps'):
  for T in (130, 512, 1127):
    l=nn.LSTM(640,256,1,batch_first=True,bidirectional=True).to(dev).eval()
    x=torch.randn(1,T,640,device=dev)
    def sync():
        if dev=='mps': torch.mps.synchronize()
    def plain():
        with torch.no_grad(): l(x); sync()
    def packed():
        with torch.no_grad():
            p=nn.utils.rnn.pack_padded_sequence(x,[T],batch_first=True,enforce_sorted=False)
            y,_=l(p); y,_=nn.utils.rnn.pad_packed_sequence(y,batch_first=True); sync()
    res={}
    for n,f in (('plain',plain),('packed',packed)):
        for _ in range(3): f()
        ts=[]
        for _ in range(7):
            t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
        res[n]=round(statistics.median(ts)*1e3,2)
    print(dev,T,res,flush=True)
