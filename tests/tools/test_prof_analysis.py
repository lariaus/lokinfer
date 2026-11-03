"""prof-lokinfer / prof-torch analysis on hand-made raw profiles (no runtime needed)."""
import copy

import pytest

from lokinfer.prof import analysis as A

# Two calls of a 3-kernel graph: a CPU kernel, then two GPU kernels (one with a wait on the in-flight segment).
RAW = {
    "version": 1, "uses_gpu": True, "num_exec_ops": 8,
    "ops": [[2, "gemm_f32_cpu", "cpu", 2, [2, 2, 2], ["f32", "f32", "f32"]],
            [4, "conv1d_lc_f32_gpu", "gpu", 3, [2, 3, 1, 2], ["f32", "f32", "f32", "f32"]],
            [6, "snake_affine_lc_f32_gpu", "gpu", 1, [2, 2], ["f32", "f32"]]],
    "calls": [
        {"call": 0, "total_ns": 1_000_000, "gpu_mode": "stream", "gpu_timestamps": True,
         "kernels": [[2, 0, 100_000, -1, -1, [4, 8, 8, 16, 4, 16]],
                     [4, 100_000, 110_000, 200_000, 500_000, [64, 128, 32, 3, 128, 32, 64, 32, 1, 1]],
                     [6, 110_000, 180_000, 500_000, 600_000, [64, 32, 64, 32], 60_000]],
         "events": [[2, "commit", 100_000, -1], [8, "finish", 180_000, 1_000_000]],
         "dispatches": [[4, "lk_conv1d_lc_f32", 2, 4, 1, 128, 1, 1, 1], [6, "lk_act_lc", 2048, 1, 1, 256, 1, 1, 0]],
         "memory": [{"op": 0, "kind": "plan", "total": 3072, "arena": 4096, "align": 256,
                     "buffers": [[1024, 0, 3, 0], [1024, 2, 5, 1024], [1000, 4, 7, 0], [1024, 6, 7, 2048]]},
                    {"op": 7, "kind": "alloc", "bytes": 512, "output": True}]},
        {"call": 1, "total_ns": 900_000, "gpu_mode": "stream", "gpu_timestamps": True,
         "kernels": [[2, 0, 120_000, -1, -1, [4, 8, 8, 16, 4, 16]],
                     [4, 120_000, 130_000, 210_000, 410_000, [64, 128, 32, 3, 128, 32, 64, 32, 1, 1]],
                     [6, 130_000, 140_000, 410_000, 520_000, [64, 32, 64, 32], 0]],
         "events": [[8, "finish", 140_000, 900_000]], "dispatches": [], "memory": []},
    ],
}

FINAL_IR = """model {
  init {
    return
  }
  exec(%a: buffer<f32[4, 8]>) {
    %arena = mem.plan sizes[1024] live[0, 3] align 256 -> M0_size, M0_o0 : buffer<u8[M0_size]>
    %v1 = view %arena[M0_o0 // 4] : buffer<f32[4, 16]>
    kernel "gemm_f32_cpu"(%a, %w) outs(%v1)
    %s = sync cpu_to_gpu %v1 : buffer<f32[4, 16]>
    kernel "conv1d_lc_f32_gpu"(%x, %w2, %b2) outs(%y)
    %v9 = view %y[0] : buffer<f32[64, 32]>
    kernel "snake_affine_lc_f32_gpu"(%v9) outs(%z)
    %out = buffer.alloc {output = true} : buffer<f32[64, 32]>
    return %z
  }
}
"""


def meta(**kw):
    return {"id": "t", "model": "M", "case": "c", "device": "Mac GPU", "target": "mac", "backend": "gpu", **kw}


def test_signature_and_split_ints():
    shapes, attrs = A.split_ints([4, 8, 8, 16, 4, 16, 1, 2], [2, 2, 2])
    assert shapes == [[4, 8], [8, 16], [4, 16]] and attrs == [1, 2]
    sig = A.signature(RAW["ops"][1], RAW["calls"][0]["kernels"][1][5])
    assert sig == "f32[64,128] f32[32,3,128] f32[32] -> f32[64,32] {1,1}"


def test_union_and_lower_bound():
    assert A.union_ns([(0, 10), (5, 20), (30, 40), (35, 36)]) == 30
    assert A.union_ns([]) == 0
    # buffers 0 and 1 overlap at ops 2..3 (2 KiB live), 1 and 2 at 4..5, 2 and 3 at 6..7
    assert A.plan_lower_bound(RAW["calls"][0]["memory"][0]["buffers"], 256) == 2048


def test_build_run_tables():
    run = A.build_run(copy.deepcopy(RAW), meta=meta(), e2e_ms=[1.0, 0.9, 1.1])
    assert run["schema"] == A.SCHEMA and run["kind"] == "lokinfer"
    k = {r["name"]: r for r in run["kernels"]}
    conv = k["conv1d_lc_f32_gpu"]
    # GPU kernels are timed by their GPU timestamps: (300 + 200) us over 2 calls
    assert conv["total_ms"] == pytest.approx(0.25)
    assert conv["calls"] == 1 and conv["mean_us"] == pytest.approx(250)
    assert conv["pct"] == pytest.approx(25.0)  # of the 1.0 ms e2e median
    assert conv["category"] == "conv" and conv["encode_us"] == pytest.approx(10)
    snake = k["snake_affine_lc_f32_gpu"]
    assert snake["wait_us"] == pytest.approx(30) and snake["encode_us"] == pytest.approx(10)
    assert snake["category"] == "activation"
    gemm = k["gemm_f32_cpu"]
    assert gemm["total_ms"] == pytest.approx(0.11) and gemm["device"] == "cpu" and "encode_us" not in gemm
    assert run["kernels"][0]["name"] == "conv1d_lc_f32_gpu"  # sorted by total time
    assert len(run["kernels_by_shape"]) == 3 and len(run["ops"]) == 3
    s = run["summary"]
    assert s["e2e_ms"]["median"] == 1.0 and s["profiled_ms"]["n"] == 2
    assert s["gpu_busy_ms"] == pytest.approx((0.4 + 0.31) / 2)
    assert s["cpu_kernel_ms"] == pytest.approx(0.11)
    assert s["inflight_wait_ms"] == pytest.approx(0.03)
    assert s["kernel_calls"] == 3 and s["gpu_kernels"] == 2
    cats = {c["category"]: c for c in run["categories"]}
    assert set(cats) == {"conv", "activation", "matmul"}


def test_timeline_memory_dispatches():
    run = A.build_run(copy.deepcopy(RAW), meta=meta(), e2e_ms=[1.0])
    tl = run["timeline"]
    assert tl["call"] == 0  # the median call: totals sorted (0.9 ms call 1, 1.0 ms call 0), index n // 2
    lanes = set(tl["lanes"])
    assert {"CPU: kernels", "CPU: encode", "GPU", "CPU: sync"} <= lanes
    gpu = [e for e in tl["events"] if e["lane"] == "GPU"]
    assert all(e["t1"] > e["t0"] for e in gpu)
    m = run["memory"]
    assert m["arena_bytes"] == 4096 and m["alloc_bytes"] == 512
    assert m["plans"][0]["lower_bound"] == 2048 and m["plans"][0]["efficiency"] == pytest.approx(2048 / 3072)
    d = {r["function"]: r for r in run["dispatches"]}
    assert d["lk_conv1d_lc_f32"]["kernels"] == ["conv1d_lc_f32_gpu"]
    assert d["lk_act_lc"]["dispatches"] == 0.5  # one dispatch logged over two profiled calls


def test_wait_shows_in_timeline():
    raw = copy.deepcopy(RAW)
    raw["calls"] = [raw["calls"][0]]
    tl = A.build_run(raw, meta=meta(), e2e_ms=[1.0])["timeline"]
    waits = [e for e in tl["events"] if e["name"] == "wait in-flight segment"]
    assert len(waits) == 1 and waits[0]["t1"] - waits[0]["t0"] == 60_000
    enc = [e for e in tl["events"] if e["lane"] == "CPU: encode" and e["op"] == 6][0]
    assert enc["t0"] == 170_000


def test_ir_helpers():
    assert A.ir_exec_line(FINAL_IR) == 4
    lines = FINAL_IR.splitlines()
    assert "gemm_f32_cpu" in lines[4 + 1 + 2] and "conv1d_lc_f32_gpu" in lines[4 + 1 + 4]
    st = A.ir_stats(FINAL_IR)
    assert st["ops"]["kernel"] == 3 and st["kernels"]["gemm_f32_cpu"] == 1
    assert st["ops"]["mem.plan"] == 1 and st["ops"]["view"] == 2 and st["ops"]["buffer.alloc"] == 1
    run = A.build_run(copy.deepcopy(RAW), meta=meta(), e2e_ms=[1.0],
                      ir_passes=[{"name": "source", "text": "x\n", "file": "ir/00.mlir"},
                                 {"name": "hoist-init", "text": FINAL_IR, "file": "ir/01.mlir"}])
    assert run["ir"]["exec_line"] == 4 and [p["name"] for p in run["ir"]["passes"]] == ["source", "hoist-init"]


@pytest.mark.parametrize("name,side,cat", [
    ("conv1d_lc_res_f32_gpu", "lokinfer", "conv"), ("convt_gather_add_lc_f32_gpu", "lokinfer", "conv_transpose"),
    ("lstm_bidir_f32_cpu", "lokinfer", "lstm"), ("gemm_f32_gpu_mps", "lokinfer", "matmul"),
    ("adain_affine_lc_f32_gpu", "lokinfer", "norm"), ("stft_magphase_f32_cpu", "lokinfer", "signal"),
    ("transpose_f32_gpu", "lokinfer", "layout"), ("attention_qkv_f32_gpu", "lokinfer", "attention"),
    ("aten::convolution", "torch", "conv"), ("aten::addmm", "torch", "matmul"), ("aten::mm", "torch", "matmul"),
    ("aten::native_layer_norm", "torch", "norm"), ("aten::_lstm_mps", "torch", "lstm"),
    ("aten::sigmoid_", "torch", "activation"), ("aten::sin", "torch", "activation"), ("aten::cat", "torch", "layout"),
    ("aten::add_", "torch", "elementwise"), ("something_unknown", "lokinfer", "other"),
])
def test_categories(name, side, cat):
    assert A.category(name, side) == cat


def test_torch_run_and_compare():
    events = []
    for call in range(2):
        events += [{"call": call, "name": "aten::convolution", "device": "cpu", "t0": 0, "t1": 400_000, "shapes": "f32[1,2]",
                    "module": "Conv1d", "category": "conv"},
                   {"call": call, "name": "aten::mul", "device": "cpu", "t0": 400_000, "t1": 450_000, "module": "LSTM",
                    "category": "lstm"},
                   {"call": call, "name": "aten::mul", "device": "cpu", "t0": 450_000, "t1": 470_000}]
    tr = A.build_torch_run(events, meta=meta(device="Mac CPU", backend="cpu"), e2e_ms=[0.5, 0.5], ncalls=2)
    names = {k["name"]: k for k in tr["kernels"]}
    assert set(names) == {"aten::convolution @conv", "aten::mul @lstm", "aten::mul"}
    assert names["aten::mul @lstm"]["category"] == "lstm" and names["aten::mul"]["category"] == "elementwise"
    assert names["aten::convolution @conv"]["total_ms"] == pytest.approx(0.4)
    assert "Conv1d: f32[1,2]" in names["aten::convolution @conv"]["shapes"]
    lk_run = A.build_run(copy.deepcopy(RAW), meta=meta(), e2e_ms=[1.0])
    c = A.compare_runs(lk_run, tr)
    assert c["e2e"]["ratio"] == pytest.approx(0.5)
    cats = {r["category"]: r for r in c["categories"]}
    assert cats["conv"]["a_ms"] == pytest.approx(0.25) and cats["conv"]["b_ms"] == pytest.approx(0.4)
    assert cats["lstm"]["a_ms"] == 0.0 and cats["lstm"]["ratio"] is None
    assert c["kernels_matched"] == 0
    same = A.compare_runs(lk_run, lk_run)
    assert same["kernels_matched"] == 3 and all(r["ratio"] == pytest.approx(1.0) for r in same["kernels"])


def test_empty_profile_rejected():
    with pytest.raises(ValueError):
        A.build_run({"ops": [], "calls": []}, meta=meta(), e2e_ms=[1.0])


def test_flops_of_matmul_like_kernels():
    from lokinfer.prof.analysis import flops_of
    assert flops_of("gemm_bias_f32_gpu", "f32[4,8] f32[8,16] f32[16] -> f32[4,16]") == 2 * 4 * 8 * 16
    assert flops_of("conv1d_lc_f32_gpu", "f32[100,64] f32[32,7,64] f32[32] -> f32[100,32] {1,3,1}") == 2 * 100 * 32 * 7 * 64
    assert flops_of("conv2d_nhwc_relu_f32_gpu", "f32[2,8,8,16] f32[32,3,3,16] f32[32] -> f32[2,8,8,32] {1,1}") == \
        2 * 2 * 8 * 8 * 32 * 3 * 3 * 16
    assert flops_of("dwconv2d_nhwc_f32_gpu", "f32[1,8,8,16] f32[3,3,16] f32[16] -> f32[1,8,8,16] {1,1}") == 2 * 64 * 9 * 16
    assert flops_of("conv1d_snake_f32_cpu", "f32[128,500] f32[128,128,7] f32[128] -> f32[128,500] f32[128,4,2]") == \
        2 * 500 * 128 * 128 * 7  # channel-first CPU conv
    assert flops_of("conv1d_snake_wino_f32_cpu", "f32[128,500] f32[64,4,128,7] -> f32[128,500]") is None
    assert flops_of("convt2d_gather_nhwc_f32_gpu", "f32[1,4,4,64] f32[4] -> f32[1,8,8,4]") is None
    assert flops_of("im2col_nchw_f32_gpu", "f32[1,3,8,8] -> f32[16,160]") is None
    assert flops_of("snake_affine_lc_f32_gpu", "f32[10,4] f32[3,4] f32[4] -> f32[10,4]") is None
