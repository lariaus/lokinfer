"""CPU (Accelerate via torch.mm) throughput for conv-as-shifted-GEMM shapes, and tile sizes; threads 8 vs 1."""
import time, statistics, torch
def bench(f, n=10):
    for _ in range(3): f()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return statistics.median(ts)
for th in (8, 1):
    torch.set_num_threads(th)
    for (M,K,N) in [(128,128,135000),(128,128,4096),(128,128*11,4096),(256,256,22540),(256,256*7,22540),(256,256,2048),(1024,1090,1127)]:
        a=torch.randn(M,K); b=torch.randn(K,N); c=torch.empty(M,N)
        t=bench(lambda: torch.mm(a,b,out=c))
        print(f'threads={th} M={M} K={K} N={N}: {t*1e3:.3f} ms {2*M*K*N/t/1e9:.0f} GFLOP/s', flush=True)
x=torch.randn(128,135000)
t=bench(lambda: torch.sin(x)); print(f'sin 128x135000: {t*1e3:.2f} ms ({x.numel()*4*2/t/1e9:.0f} GB/s eff)')
