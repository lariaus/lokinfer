import time, statistics, torch, numpy as np
print(torch.__config__.show().split('\n')[:12])
def bench(f, sync, n=10):
    for _ in range(3): f(); sync()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); sync(); ts.append(time.perf_counter()-t)
    return statistics.median(ts)
for dev in ('cpu','mps'):
    sync=(lambda: torch.mps.synchronize()) if dev=='mps' else (lambda: None)
    for (M,K,N,tag) in [(2048,2048,2048,'square'),(128,128*11,135000,'res5 conv im2col (F=1127)'),(256,256*7,22540,'stage0 k7'),(1024,1090*3,1127,'decode k3')]:
        a=torch.randn(M,K,device=dev); b=torch.randn(K,N,device=dev)
        t=bench(lambda: a@b, sync)
        print(dev, tag, (M,K,N), f'{t*1e3:.2f} ms', f'{2*M*K*N/t/1e9:.0f} GFLOP/s', flush=True)
    # actual conv1d
    for (C,k,L,d) in [(128,11,135241,1),(128,11,135241,5),(256,7,22540,3)]:
        conv=torch.nn.Conv1d(C,C,k,dilation=d,padding=d*(k-1)//2).to(dev)
        x=torch.randn(1,C,L,device=dev)
        with torch.no_grad():
            t=bench(lambda: conv(x), sync)
        print(dev, f'conv1d C={C} k={k} d={d} L={L}', f'{t*1e3:.2f} ms', f'{2*C*C*k*L/t/1e9:.0f} GFLOP/s', flush=True)
