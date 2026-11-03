"""Kokoro CPU kernels (kokoro-port-plan.md step 5) vs PyTorch functional ops / the kokoro modules.

Tolerances (f32 kernels, O(1) data unless noted):
  * pure data movement (embedding, concat, broadcast, repeat_interleave, upsample, pad): bit-exact
  * elementwise (gelu, leaky, scale, add3, snake): rtol 1e-6 / atol 1e-6
  * norms: atol 2e-5 vs a float64 reference
  * GEMM-based (conv, conv_transpose, attention, LSTM): float64 reference, atol/rtol 1e-4 (LSTM 1e-4 over
    up to 130 steps)
  * durations: exact (float-sum reorderings can only flip values within 1e-4 of x.5; none occur here)
  * harmonic source: phase computed in torch's exact op order (bit-exact phase); output max abs error
    vs the kokoro module <= 2e-6 (Linear-sum order and libm sin differences only)
  * STFT: magnitude atol 1e-5 * max|X|; angle wrapped diff atol 1e-4 where |X| > 1e-3 * max|X|
  * iSTFT: atol 1e-5 * max|y|
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import lokinfer as lk
from lokinfer import ops

torch.set_num_threads(1)


def T(a, dtype=np.float32):
    return lk.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=dtype)), copy=True)


def rng(seed):
    return np.random.default_rng(seed)


def randn(r, *shape, scale=1.0):
    return (r.standard_normal(shape) * scale).astype(np.float32)


def t64(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def run(fn, *inputs, **kw):
    model = lk.compile(fn, inputs=list(inputs), **kw)
    out = model(*inputs)
    return [o.numpy() for o in out] if isinstance(out, tuple) else out.numpy()


# --- token-rate -----------------------------------------------------------------------------------


@pytest.mark.parametrize("t,d", [(1, 128), (7, 512), (512, 128), (3, 1)])
def test_embedding(t, d):
    r = rng(t * 1000 + d)
    table = randn(r, 178, d)
    ids = r.integers(0, 178, t).astype(np.int64)
    out = ops.embedding_f32_cpu(T(table), T(ids, np.int64)).numpy()
    np.testing.assert_array_equal(out, table[ids])


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 128), (7, 768), (130, 512), (3, 5), (2, 4099)])
@pytest.mark.parametrize("eps", [1e-12, 1e-5])
def test_layernorm(rows, cols, eps):
    r = rng(rows * 7 + cols)
    x = randn(r, rows, cols) * 3 + 50  # offset mean: exercises the two-pass stats
    g, b = randn(r, cols), randn(r, cols)
    out = ops.layernorm_f32_cpu(T(x), T(g), T(b), eps=eps).numpy()
    ref = F.layer_norm(t64(x), (cols,), t64(g), t64(b), eps).numpy()
    np.testing.assert_allclose(out, ref, atol=2e-5 * max(1.0, np.abs(ref).max()), rtol=0)


@pytest.mark.parametrize("rows,cols", [(1, 512), (7, 512), (130, 512), (5, 3)])
def test_adalayernorm(rows, cols):
    r = rng(rows + cols)
    x = randn(r, rows, cols) + 2
    gb = randn(r, 2 * cols, scale=0.5)
    out = ops.adalayernorm_f32_cpu(T(x), T(gb), eps=1e-5).numpy()
    g, b = t64(gb[:cols]), t64(gb[cols:])
    ref = ((1 + g) * F.layer_norm(t64(x), (cols,), eps=1e-5) + b).numpy()
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=0)


@pytest.mark.parametrize("t", [1, 7, 130, 512])
def test_attention(t):
    heads, hd = 12, 64
    r = rng(t)
    q, k, v = (randn(r, t, heads * hd) for _ in range(3))
    out = ops.attention_f32_cpu(T(q), T(k), T(v), heads=heads).numpy()

    def split(a):
        return t64(a).view(t, heads, hd).transpose(0, 1)[None]

    ref = F.scaled_dot_product_attention(split(q), split(k), split(v))[0].transpose(0, 1).reshape(t, heads * hd)
    np.testing.assert_allclose(out, ref.numpy(), atol=1e-4, rtol=1e-4)


def test_attention_other_head_dim():
    t, heads, hd = 9, 3, 5
    r = rng(5)
    q, k, v = (randn(r, t, heads * hd) for _ in range(3))
    out = ops.attention_f32_cpu(T(q), T(k), T(v), heads=heads).numpy()

    def split(a):
        return t64(a).view(t, heads, hd).transpose(0, 1)[None]

    ref = F.scaled_dot_product_attention(split(q), split(k), split(v))[0].transpose(0, 1).reshape(t, heads * hd)
    np.testing.assert_allclose(out, ref.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("t,i,h", [(1, 3, 2), (7, 5, 4), (2, 1, 1), (13, 640, 256), (30, 512, 256), (130, 640, 256)])
def test_lstm_bidir(t, i, h):
    torch.manual_seed(t * 31 + i)
    lstm = torch.nn.LSTM(i, h, 1, batch_first=True, bidirectional=True).double()
    x = randn(rng(t + i), t, i)
    with torch.no_grad():
        ref = lstm(t64(x)[None])[0][0].numpy()
    p = {k: v.detach().numpy().astype(np.float32) for k, v in lstm.named_parameters()}
    w_ih = np.concatenate([p["weight_ih_l0"], p["weight_ih_l0_reverse"]])
    w_hh = np.concatenate([p["weight_hh_l0"], p["weight_hh_l0_reverse"]])
    b = np.concatenate([p["bias_ih_l0"] + p["bias_hh_l0"], p["bias_ih_l0_reverse"] + p["bias_hh_l0_reverse"]])
    out = ops.lstm_bidir_f32_cpu(T(x), T(w_ih), T(w_hh), T(b)).numpy()
    assert out.shape == (t, 2 * h)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def _lstm_params(i, h, seed):
    torch.manual_seed(seed)
    lstm = torch.nn.LSTM(i, h, 1, batch_first=True, bidirectional=True).double()
    p = {k: v.detach().numpy().astype(np.float32) for k, v in lstm.named_parameters()}
    w_ih = np.concatenate([p["weight_ih_l0"], p["weight_ih_l0_reverse"]])
    w_hh = np.concatenate([p["weight_hh_l0"], p["weight_hh_l0_reverse"]])
    b = np.concatenate([p["bias_ih_l0"] + p["bias_hh_l0"], p["bias_ih_l0_reverse"] + p["bias_hh_l0_reverse"]])
    return lstm, w_ih, w_hh, b


@pytest.mark.parametrize("t1,i1,t2,i2,h", [(59, 640, 59, 512, 256), (3, 5, 9, 7, 4), (1, 3, 40, 3, 8), (130, 640, 2, 512, 256)])
def test_lstm_bidir2_equals_two_calls(t1, i1, t2, i2, h):
    """lstm_bidir2 (two BiLSTMs, 4 recurrences concurrently) == two lstm_bidir calls, bit for bit."""
    _, wi1, wh1, b1 = _lstm_params(i1, h, 1)
    _, wi2, wh2, b2 = _lstm_params(i2, h, 2)
    x1, x2 = randn(rng(t1), t1, i1), randn(rng(t2 + 7), t2, i2)
    o1, o2 = run(lambda a, c: ops.lstm_bidir2_f32_cpu(a, T(wi1), T(wh1), T(b1), c, T(wi2), T(wh2), T(b2)), T(x1), T(x2))
    np.testing.assert_array_equal(o1, ops.lstm_bidir_f32_cpu(T(x1), T(wi1), T(wh1), T(b1)).numpy())
    np.testing.assert_array_equal(o2, ops.lstm_bidir_f32_cpu(T(x2), T(wi2), T(wh2), T(b2)).numpy())


@pytest.mark.parametrize("t,i,h", [(7, 5, 4), (13, 640, 256), (130, 640, 256)])
def test_lstm_bidir_t(t, i, h):
    """The transposed-w_hh BiLSTM (SME2 / NEON 4-partial GEMV) vs float64 torch."""
    lstm, w_ih, w_hh, b = _lstm_params(i, h, t * 31 + i)
    x = randn(rng(t + i), t, i)
    with torch.no_grad():
        ref = lstm(t64(x)[None])[0][0].numpy()
    out = ops.lstm_bidir_t_f32_cpu(T(x), T(w_ih), T(np.ascontiguousarray(w_hh.T)), T(b)).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def _torch_durations(x, speed):
    d = torch.sigmoid(torch.from_numpy(x)).sum(axis=-1) / speed
    return d.numpy(), torch.round(d).clamp(min=1).long().numpy()


def _durations(x, speed):
    return run(lambda a, s: ops.durations_f32_cpu(a, s), T(x), float(speed))


def test_durations_round_half_even_and_clamp():
    # sigmoid(0) = 0.5 and sigmoid(-1000) = 0 exactly, so the per-row sums are exact multiples of 0.5.
    zeros = [5, 7, 1, 0, 3, 50, 9, 2]  # sums 2.5 3.5 0.5 0 1.5 25 4.5 1.0
    x = np.full((len(zeros), 50), -1000.0, np.float32)
    for row, z in enumerate(zeros):
        x[row, :z] = 0.0
    dur, total = _durations(x, 1.0)
    expect = np.array([2, 4, 1, 1, 2, 25, 4, 1], np.int64)
    np.testing.assert_array_equal(dur, expect)
    np.testing.assert_array_equal(dur, _torch_durations(x, 1.0)[1])
    assert total.tolist() == [int(expect.sum())]
    dur2, _ = _durations(x, 2.0)  # 1.25 1.75 0.25 0 0.75 12.5 2.25 0.5
    np.testing.assert_array_equal(dur2, [1, 2, 1, 1, 1, 12, 2, 1])
    np.testing.assert_array_equal(dur2, _torch_durations(x, 2.0)[1])


@pytest.mark.parametrize("t", [1, 3, 30, 512])
@pytest.mark.parametrize("speed", [0.8, 1.0, 1.25])
def test_durations_random(t, speed):
    x = randn(rng(t), t, 50, scale=3) - 2
    dur, total = _durations(x, speed)
    ref_f, ref = _torch_durations(x, np.float32(speed))
    near_half = np.abs(ref_f - np.floor(ref_f) - 0.5) < 1e-4
    assert not np.any((dur != ref) & ~near_half)
    np.testing.assert_array_equal(dur[~near_half], ref[~near_half])
    assert total.tolist() == [int(dur.sum())]


def test_repeat_interleave_from_durations_input():
    """Forced-durations variant: dur and total are inputs, F = dim_read(total)."""
    Tn, C = lk.Sym("T"), lk.Sym("C")

    def fn(x, dur, total):
        f = lk.dim_read(total)
        return ops.repeat_interleave_f32_cpu(x, dur, frames=f)

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [Tn, C]), lk.TensorType(lk.i64, [Tn]),
                                        lk.TensorType(lk.i64, [1])])
    for t, c, seed in [(1, 1, 0), (5, 3, 1), (130, 512, 2)]:
        r = rng(seed)
        x = randn(r, t, c)
        dur = r.integers(1, 9, t).astype(np.int64)
        out = model(T(x), T(dur, np.int64), T([dur.sum()], np.int64)).numpy()
        ref = torch.from_numpy(x).repeat_interleave(torch.from_numpy(dur), dim=0).numpy()
        np.testing.assert_array_equal(out, ref)
    # Frames that don't match the durations are an error, never truncation or zero padding.
    x, dur = randn(rng(3), 4, 2), np.array([1, 2, 3, 4], np.int64)
    for wrong in (9, 11, 1):
        with pytest.raises(RuntimeError, match="don't sum to the frame count"):
            model(T(x), T(dur, np.int64), T([wrong], np.int64))


def test_embedding_rejects_out_of_range_ids():
    table = randn(rng(4), 10, 3)
    model = lk.compile(lambda t, i: ops.embedding_f32_cpu(t, i),
                       input_types=[lk.TensorType(lk.f32, [10, 3]), lk.TensorType(lk.i64, [lk.Sym("T")])])
    np.testing.assert_array_equal(model(T(table), T([0, 9], np.int64)).numpy(), table[[0, 9]])
    for bad in ([10], [-1], [3, 100]):
        with pytest.raises(RuntimeError, match="out of range"):
            model(T(table), T(bad, np.int64))


def test_durations_dim_read_repeat_interleave_e2e():
    """durations -> dim.read -> repeat_interleave, one compiled graph, F changes from run to run."""
    Tn = lk.Sym("T")

    def fn(logits, speed, h):
        dur, total = ops.durations_f32_cpu(logits, speed)
        f = lk.dim_read(total, 0, name="F")
        return ops.repeat_interleave_f32_cpu(h, dur, frames=f), dur

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [Tn, 50]), lk.ScalarType(lk.f32),
                                        lk.TensorType(lk.f32, [Tn, 640])])
    assert "dim.read" in model.source_ir
    for t, speed, seed in [(3, 1.0, 0), (16, 0.8, 1), (119, 1.25, 2), (16, 1.0, 3), (512, 1.0, 4)]:
        r = rng(seed)
        logits = randn(r, t, 50, scale=3) - 1
        h = randn(r, t, 640)
        out, dur = model(T(logits), float(speed), T(h))
        out, dur = out.numpy(), dur.numpy()
        _, ref_dur = _torch_durations(logits, np.float32(speed))
        np.testing.assert_array_equal(dur, ref_dur)
        ref = torch.from_numpy(h).repeat_interleave(torch.from_numpy(ref_dur), dim=0).numpy()
        assert out.shape == (int(ref_dur.sum()), 640)
        np.testing.assert_array_equal(out, ref)


def test_dim_read_needs_tracing_and_int_tensor():
    with pytest.raises(ops.OpError):
        lk.dim_read(T([3], np.int64))
    with pytest.raises(ops.OpError, match="i32/i64"):
        lk.compile(lambda a: ops.broadcast_rows_f32_cpu(a, lk.dim_read(a)), inputs=[T([1.0])])


@pytest.mark.parametrize("rows,cols", [(1, 1), (5, 128), (3, 7)])
def test_broadcast_rows(rows, cols):
    v = randn(rng(cols), cols)
    out = ops.broadcast_rows_f32_cpu(T(v), rows=rows).numpy()
    np.testing.assert_array_equal(out, np.broadcast_to(v, (rows, cols)))


def test_broadcast_rows_symbolic():
    Tn = lk.Sym("T")

    def fn(x, v):  # cat([x, s_p broadcast]) as in the DurationEncoder
        return ops.concat_cols_f32_cpu(x, ops.broadcast_rows_f32_cpu(v, rows=x.shape[0]))

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [Tn, 512]), lk.TensorType(lk.f32, [128])])
    for t in [1, 17]:
        r = rng(t)
        x, v = randn(r, t, 512), randn(r, 128)
        np.testing.assert_array_equal(model(T(x), T(v)).numpy(),
                                      np.concatenate([x, np.broadcast_to(v, (t, 128))], axis=1))


@pytest.mark.parametrize("sizes", [[1, 1], [3, 5, 7], [2, 1, 4, 1, 3, 6, 2, 9]])
def test_concat_rows_cols(sizes):
    r = rng(len(sizes))
    xs = [randn(r, s, 13) for s in sizes]
    out = ops.concat_rows_f32_cpu(*[T(x) for x in xs]).numpy()
    np.testing.assert_array_equal(out, np.concatenate(xs, axis=0))
    ys = [randn(r, 11, s) for s in sizes]
    out = ops.concat_cols_f32_cpu(*[T(y) for y in ys]).numpy()
    np.testing.assert_array_equal(out, np.concatenate(ys, axis=1))


def test_concat_arity_checked():
    x = T(np.zeros((2, 2)))
    with pytest.raises(ops.OpError, match="2..8"):
        ops.concat_rows_f32_cpu(x)
    with pytest.raises(ops.OpError, match="2..8"):
        ops.concat_cols_f32_cpu(*([x] * 9))


# --- elementwise ----------------------------------------------------------------------------------

SHAPES = [(1,), (7,), (5, 3), (130, 2048), (2, 3, 5)]


@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_tanh(shape):
    x = randn(rng(len(shape)), *shape, scale=3)
    out = ops.gelu_tanh_f32_cpu(T(x)).numpy()
    np.testing.assert_allclose(out, F.gelu(torch.from_numpy(x), approximate="tanh").numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("slope", [0.2, 0.1, 0.01])
def test_leaky_relu(shape, slope):
    x = randn(rng(len(shape)), *shape)
    out = ops.leaky_relu_f32_cpu(T(x), slope).numpy()
    np.testing.assert_array_equal(out, F.leaky_relu(torch.from_numpy(x), slope).numpy())


@pytest.mark.parametrize("shape", SHAPES)
def test_scale_add3(shape):
    r = rng(sum(shape))
    a, b, c = randn(r, *shape), randn(r, *shape), randn(r, *shape)
    s = 1 / math.sqrt(2)
    np.testing.assert_array_equal(ops.scale_f32_cpu(T(a), s).numpy(), a * np.float32(s))
    out = ops.add3_scale_f32_cpu(T(a), T(b), T(c), 1 / 3).numpy()
    ref = ((torch.from_numpy(a) + torch.from_numpy(b) + torch.from_numpy(c)) * (1 / 3)).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("c,l", [(1, 1), (7, 13), (128, 1001)])
def test_snake(c, l):
    r = rng(c + l)
    x = randn(r, c, l, scale=2)
    alpha = (r.uniform(0.2, 3.0, c)).astype(np.float32)
    out = ops.snake_f32_cpu(T(x), T(alpha)).numpy()
    xt, a = torch.from_numpy(x)[None], torch.from_numpy(alpha)[None, :, None]
    ref = (xt + (1 / a) * (torch.sin(a * xt) ** 2))[0].numpy()  # kokoro AdaINResBlock1 expression
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


# --- frame-rate convs -----------------------------------------------------------------------------

# (Ci, Co, L, K, stride, padding, dilation): every combination Kokoro uses plus edge cases.
CONVS = [
    (16, 24, 50, 3, 1, 1, 1),
    (16, 16, 50, 3, 1, 3, 3),
    (16, 16, 50, 3, 1, 5, 5),
    (13, 7, 33, 5, 1, 2, 1),
    (8, 8, 41, 7, 1, 3, 1),
    (8, 8, 41, 7, 1, 9, 3),
    (8, 8, 41, 7, 1, 15, 5),
    (8, 8, 61, 11, 1, 5, 1),
    (8, 8, 61, 11, 1, 15, 3),
    (8, 8, 61, 11, 1, 25, 5),
    (514, 31, 17, 1, 1, 0, 1),  # 1x1 (pointwise GEMM path)
    (22, 32, 1201, 12, 6, 3, 1),  # noise_convs[0]
    (1, 1, 334, 3, 2, 1, 1),  # F0_conv / N_conv
    (1, 1, 1, 3, 2, 1, 1),
    (5, 3, 1, 3, 1, 1, 1),  # L = 1
    (4, 3, 2, 7, 1, 3, 1),  # K > L
    (3, 2, 3, 11, 1, 25, 5),  # dilated K*d > L
    (256, 32, 2000, 11, 1, 25, 5),  # several im2col column tiles
    (22, 16, 60001, 12, 6, 3, 1),  # strided, several tiles
    (6, 5, 20, 4, 3, 2, 2),  # stride + dilation together
]


@pytest.mark.parametrize("ci,co,l,k,s,p,d", CONVS)
def test_conv1d(ci, co, l, k, s, p, d):
    r = rng(ci * 7 + l)
    x = randn(r, ci, l)
    w = randn(r, co, ci, k, scale=1 / math.sqrt(ci * k))
    b = randn(r, co)
    out = ops.conv1d_f32_cpu(T(x), T(w), T(b), stride=s, padding=p, dilation=d).numpy()
    ref = F.conv1d(t64(x)[None], t64(w), t64(b), stride=s, padding=p, dilation=d)[0].numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
    out = ops.conv1d_nobias_f32_cpu(T(x), T(w), stride=s, padding=p, dilation=d).numpy()
    ref = F.conv1d(t64(x)[None], t64(w), stride=s, padding=p, dilation=d)[0].numpy()
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_conv1d_symbolic_length():
    L = lk.Sym("L")
    r = rng(0)
    w, b = randn(r, 8, 6, 12, scale=0.1), randn(r, 8)
    model = lk.compile(lambda x: ops.conv1d_f32_cpu(x, T(w), T(b), stride=6, padding=3),
                       input_types=[lk.TensorType(lk.f32, [6, L])])
    for l in [7, 121, 1201]:
        x = randn(r, 6, l)
        ref = F.conv1d(t64(x)[None], t64(w), t64(b), stride=6, padding=3)[0].numpy()
        np.testing.assert_allclose(model(T(x)).numpy(), ref, atol=1e-4, rtol=1e-4)


def test_conv1d_empty_output_rejected():
    with pytest.raises(ops.OpError, match="empty"):
        ops.conv1d_f32_cpu(T(np.zeros((2, 3))), T(np.zeros((2, 2, 7))), T(np.zeros(2)))


# (Ci, Co, L, K, stride, padding, output_padding)
CONVTS = [
    (64, 32, 37, 20, 10, 5, 0),  # ups[0]
    (32, 16, 50, 12, 6, 3, 0),  # ups[1]
    (5, 3, 1, 20, 10, 5, 0),  # L = 1
    (7, 9, 13, 3, 2, 1, 1),
    (32, 256, 1000, 20, 10, 5, 0),  # several col2im tiles
    (4, 6, 9, 5, 1, 2, 0),
    (3, 2, 4, 2, 3, 0, 2),  # K < stride (gaps) + output_padding
]


@pytest.mark.parametrize("ci,co,l,k,s,p,op", CONVTS)
def test_conv_transpose1d(ci, co, l, k, s, p, op):
    r = rng(ci + co + l)
    x = randn(r, ci, l)
    w = randn(r, ci, co, k, scale=1 / math.sqrt(ci * k / s))
    b = randn(r, co)
    out = ops.conv_transpose1d_f32_cpu(T(x), T(w), T(b), stride=s, padding=p, output_padding=op).numpy()
    ref = F.conv_transpose1d(t64(x)[None], t64(w), t64(b), stride=s, padding=p, output_padding=op)[0].numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("c,l", [(1, 1), (7, 5), (64, 334), (1090, 31)])
def test_conv_transpose1d_dw(c, l):
    r = rng(c + l)
    x, w, b = randn(r, c, l), randn(r, c, 1, 3), randn(r, c)
    out = ops.conv_transpose1d_dw_f32_cpu(T(x), T(w), T(b), stride=2, padding=1, output_padding=1).numpy()
    ref = F.conv_transpose1d(t64(x)[None], t64(w), t64(b), stride=2, padding=1, output_padding=1, groups=c)[0]
    assert out.shape == (c, 2 * l)
    np.testing.assert_allclose(out, ref.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("c,l", [(1, 1), (7, 13), (512, 334), (16, 135001)])
def test_adain1d(c, l):
    r = rng(c + l)
    x = randn(r, c, l, scale=4) + 100  # large mean, long rows (research trap #4)
    gb = randn(r, 2 * c, scale=0.5)
    out = ops.adain1d_f32_cpu(T(x), T(gb), eps=1e-5).numpy()
    g, b = t64(gb[:c])[:, None], t64(gb[c:])[:, None]
    if l > 1:
        xn = F.instance_norm(t64(x)[None], eps=1e-5)[0]
    else:  # torch refuses L = 1 (single spatial element); the normalized value is 0
        xn = torch.zeros(c, 1, dtype=torch.float64)
    ref = ((1 + g) * xn + b).numpy()
    np.testing.assert_allclose(out, ref, atol=2e-5 * max(1.0, np.abs(ref).max()), rtol=0)


@pytest.mark.parametrize("c,l", [(1, 1), (3, 5), (256, 334)])
@pytest.mark.parametrize("scale", [2, 3, 300])
def test_upsample_nearest1d(c, l, scale):
    x = randn(rng(l), c, l)
    out = ops.upsample_nearest1d_f32_cpu(T(x), scale).numpy()
    np.testing.assert_array_equal(out, F.interpolate(torch.from_numpy(x)[None], scale_factor=scale)[0].numpy())


@pytest.mark.parametrize("c,l,left,right", [(1, 2, 1, 0), (128, 40, 1, 0), (3, 11, 10, 10), (2, 5, 3, 2), (4, 21, 0, 4)])
def test_pad_reflect1d(c, l, left, right):
    x = randn(rng(l), c, l)
    out = ops.pad_reflect1d_f32_cpu(T(x), left, right).numpy()
    np.testing.assert_array_equal(out, F.pad(torch.from_numpy(x)[None], (left, right), mode="reflect")[0].numpy())


def test_pad_reflect1d_too_wide_rejected():
    with pytest.raises(ops.OpError, match="reflect"):
        ops.pad_reflect1d_f32_cpu(T(np.zeros((1, 3))), 3, 0)


# --- signal (vs the kokoro modules) ---------------------------------------------------------------

istftnet = pytest.importorskip("kokoro.istftnet")


def f0_curve(n, seed):
    """Realistic F0 at 80 Hz: smooth 70..400 Hz contours, unvoiced (0) gaps, sub-threshold values."""
    r = rng(seed)
    t = np.arange(n)
    f0 = 180 + 90 * np.sin(t / 17.0 + seed) + 40 * np.sin(t / 5.3) + r.normal(0, 5, n)
    f0 = np.clip(f0, 70, 400)
    for _ in range(max(1, n // 40)):
        a = r.integers(0, n)
        f0[a:a + r.integers(2, 12)] = 0.0
    f0[r.integers(0, n, max(1, n // 50))] = r.uniform(0, 10)  # unvoiced by threshold (f0 <= 10)
    f0[0] = 0.0
    return f0.astype(np.float32)


def torch_harmonic_source(f0, noise, w, b, monkeypatch):
    m = istftnet.SourceModuleHnNSF(sampling_rate=24000, upsample_scale=300, harmonic_num=8, voiced_threshod=10)
    with torch.no_grad():
        m.l_linear.weight.copy_(torch.from_numpy(w)[None])
        m.l_linear.bias.copy_(torch.from_numpy(b))
    draws = iter([torch.from_numpy(noise)[None], None])

    def fake_randn_like(x, *a, **k):
        v = next(draws)
        return v if v is not None else torch.zeros_like(x)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    f0u = torch.nn.Upsample(scale_factor=300)(torch.from_numpy(f0)[None, None]).transpose(1, 2)
    with torch.no_grad():
        har, _, _ = m(f0u)
    monkeypatch.undo()
    return har[0, :, 0].numpy()


@pytest.mark.parametrize("frames", [1, 3, 30, 119, 300])
def test_harmonic_source(frames, monkeypatch):
    n = 2 * frames
    r = rng(frames)
    f0 = f0_curve(n, frames)
    noise = randn(r, 300 * n, 9)
    w, b = randn(r, 9, scale=0.5), randn(r, 1, scale=0.1)
    out = ops.harmonic_source_f32_cpu(T(f0), T(noise), T(w), T(b)).numpy()
    ref = torch_harmonic_source(f0, noise, w, b, monkeypatch)
    assert out.shape == ref.shape == (600 * frames,)
    err = np.abs(out - ref)
    snr = 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum(err.astype(np.float64) ** 2), 1e-300))
    assert err.max() <= 2e-6, (err.max(), snr)


def torch_stft():
    return istftnet.TorchSTFT(filter_length=20, hop_length=5, win_length=20)


@pytest.mark.parametrize("s", [11, 25, 600, 6001, 600 * 300])
def test_stft_magphase(s):
    r = rng(s)
    t = np.arange(s)
    x = (0.3 * np.sin(t * 0.05) + 0.2 * np.sin(t * 0.9) + 0.05 * r.standard_normal(s)).astype(np.float32)
    x[: s // 7] = 0.0  # silent stretch: DC/Nyquist bins with imag == 0
    out = ops.stft_magphase_f32_cpu(T(x), n_fft=20, hop=5).numpy()
    mag, ph = torch_stft().transform(torch.from_numpy(x)[None])
    mag, ph = mag[0].numpy(), ph[0].numpy()
    assert out.shape == (22, s // 5 + 1)
    np.testing.assert_allclose(out[:11], mag, atol=1e-5 * mag.max(), rtol=0)
    sig = mag > 1e-3 * mag.max()
    dph = np.angle(np.exp(1j * (out[11:].astype(np.float64) - ph)))  # wrapped: +-pi branch cut is not an error
    assert np.abs(dph[sig]).max() < 1e-4


@pytest.mark.parametrize("l", [2, 3, 41, 1201, 120 * 300 + 1])
def test_istft_logits(l):
    r = rng(l)
    x = randn(r, 22, l, scale=0.7) - np.float32(1.0)
    out = ops.istft_logits_f32_cpu(T(x), n_fft=20, hop=5).numpy()
    xt = torch.from_numpy(x)[None]
    ref = torch_stft().inverse(torch.exp(xt[:, :11]), torch.sin(xt[:, 11:]))[0, 0].numpy()
    assert out.shape == ref.shape == (5 * (l - 1),)
    np.testing.assert_allclose(out, ref, atol=1e-5 * np.abs(ref).max(), rtol=0)


def test_stft_of_harmonic_source_chain(monkeypatch):
    """har = STFT(harmonic_source(F0)) as Generator.forward computes it, F = 119 (real117)."""
    frames = 119
    n = 2 * frames
    r = rng(117)
    f0, noise = f0_curve(n, 117), randn(r, 300 * n, 9)
    w, b = randn(r, 9, scale=0.5), randn(r, 1, scale=0.1)

    def fn(f, z):
        return ops.stft_magphase_f32_cpu(ops.harmonic_source_f32_cpu(f, z, T(w), T(b)))

    out = run(fn, T(f0), T(noise))
    src = torch_harmonic_source(f0, noise, w, b, monkeypatch)
    mag, ph = torch_stft().transform(torch.from_numpy(src)[None])
    mag, ph = mag[0].numpy(), ph[0].numpy()
    assert out.shape == (22, 120 * frames + 1)
    np.testing.assert_allclose(out[:11], mag, atol=1e-5 * mag.max(), rtol=0)
    sig = mag > 1e-3 * mag.max()
    dph = np.angle(np.exp(1j * (out[11:].astype(np.float64) - ph)))
    assert np.abs(dph[sig]).max() < 1e-3
