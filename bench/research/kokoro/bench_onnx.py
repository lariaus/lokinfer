"""Benchmark onnx-community Kokoro-82M-v1.0 ONNX (fp32) with onnxruntime CPU / CoreML EP.
usage: python -I bench_onnx.py MODEL.onnx INPUTS.npz EP [runs] [warmup]
EP: cpu | coreml_all | coreml_cpugpu | coreml_cpu
INPUTS.npz: produced by dump_inputs.py (ids_<case>, style_<case>)."""
import json
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

model, inputs, ep = sys.argv[1], sys.argv[2], sys.argv[3]
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 2

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if ep == 'cpu':
    providers = ['CPUExecutionProvider']
else:
    units = {'coreml_all': 'ALL', 'coreml_cpugpu': 'CPUAndGPU', 'coreml_cpu': 'CPUOnly'}[ep]
    providers = [('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': units,
                                              'RequireStaticInputShapes': '0'}), 'CPUExecutionProvider']
t0 = time.perf_counter()
sess = ort.InferenceSession(model, so, providers=providers)
print(json.dumps({'ep': ep, 'ort': ort.__version__, 'load_s': time.perf_counter() - t0,
                  'providers': sess.get_providers(),
                  'inputs': [(i.name, i.type, i.shape) for i in sess.get_inputs()]}), flush=True)
d = np.load(inputs)
cases = sorted({k.split('_', 1)[1] for k in d.files}, key=lambda c: d['ids_' + c].shape[1])
for c in cases:
    feeds = {'input_ids': d['ids_' + c].astype(np.int64), 'style': d['style_' + c].astype(np.float32),
             'speed': np.ones(1, dtype=np.float32)}
    for _ in range(warmup):
        out = sess.run(None, feeds)[0]
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = sess.run(None, feeds)[0]
        ts.append(time.perf_counter() - t)
    print(json.dumps({'case': c, 'T': int(feeds['input_ids'].shape[1]), 'samples': int(out.size),
                      'median_ms': statistics.median(ts) * 1e3, 'min_ms': min(ts) * 1e3}), flush=True)
