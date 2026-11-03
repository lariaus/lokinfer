"""Kokoro-82M: lokinfer graph vs the PyTorch reference (models/kokoro/reference.py),
stage by stage and end to end (docs/impl-notes/kokoro-port-plan.md)."""
import numpy as np
import pytest

pytest.importorskip("kokoro")

from models.kokoro import model as K  # noqa: E402
from models.kokoro import reference as R  # noqa: E402

# The app's real workload (models/kokoro/reference.py REAL_TEXTS): real sentences at the p5 / median / p95 token
# length in both app voices (f = af_heart, m = am_echo), a one-word utterance, and the median sentence in three more
# voices (slowest, British, fast).
APP = ["word", "app47f", "app47m", "app59f", "app59m", "app71f", "app71m", "app59_nicole", "app59_emma", "app59_fenrir"]
# Longer utterances: 6 s / 8 s sentences in both voices, a 20 s paragraph in both voices, a real-text near-max chunk.
LONG = ["s6f", "s6m", "s9f", "s9m", "s20f", "s20m", "s28f"]

STAGES = ("bert_dur", "d", "lstm_x", "dur_logits", "en", "shared", "F0_pred", "N_pred", "t_en", "asr", "dec_x",
          "har_source", "har", "post", "audio")


@pytest.fixture(scope="module")
def weights():
    return K.fold_weights(R.load_model())


def inputs(c, forced: bool):
    third = c["pred_dur"].astype(np.int64) if forced else np.float32(c["speed"])
    return [np.concatenate([[0], c["ids"], [0]]).astype(np.int64), c["ref_s"], third, c["noise"]]


def rel_err(got, exp):
    return float(np.abs(got - exp).max() / max(np.abs(exp).max(), 1e-12))


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
@pytest.mark.parametrize("case", ["rand1", "real28", "app59f", "app59m", "s9m", "rand128"])
def test_stages_forced_durations(weights, case, gpu):
    """Every stage vs the PyTorch reference (forced durations). gpu=True: ALBERT + decoder + generator on the GPU
    (M4), judged by the same thresholds as the CPU graph."""
    c = R.load_case(case)
    model = K.build(weights, forced_durations=True, explicit_noise=True, outputs=STAGES, gpu=gpu)
    outs = model(*inputs(c, forced=True))
    report = {}
    for name, got in zip(STAGES, outs):
        got = got.numpy()
        exp = c[name]
        assert got.shape == exp.shape, (name, got.shape, exp.shape)
        report[name] = rel_err(got, exp)
    print(case, {k: f"{v:.1e}" for k, v in report.items()})
    # Token-rate stages are tight; frame-rate stages accumulate f32 reordering
    # (and the SineGen phase is ~1e5 rad, research-kokoro.md §2.5 #2).
    for name in ("bert_dur", "d", "lstm_x", "dur_logits", "en", "t_en", "asr"):
        assert report[name] < 1e-4, (name, report[name])
    # `shared` (BiLSTM over ~130 frames at rand128) amplifies ulp-level input differences: swapping GELU's tanh
    # (std::tanh vs vvtanhf; BERT output moves by ~1e-6) moved it between 5.9e-5 and 1.2e-4 while other stages
    # moved the opposite way (2026-11-01). Bound it at the measured spread x2.
    assert report["shared"] < 2.5e-4, ("shared", report["shared"])
    for name in ("F0_pred", "N_pred", "dec_x", "post"):
        assert report[name] < 1e-2, (name, report[name])
    # har = [|X|; angle(X)]: magnitudes compared directly, angles circularly
    # (the ±π branch cut flips where imag ≈ 0, research-kokoro.md §2.5 #3).
    got_har, exp_har = outs[STAGES.index("har")].numpy(), c["har"]
    mag = rel_err(got_har[:11], exp_har[:11])
    dphi = np.angle(np.exp(1j * (got_har[11:].astype(np.float64) - exp_har[11:])))
    weight = exp_har[:11] / max(exp_har[:11].max(), 1e-12)  # phase only matters where there is energy
    print(case, f"har |X| rel {mag:.1e}, energy-weighted phase err {float(np.abs(dphi * weight).max()):.1e} rad")
    assert mag < 5e-2, mag
    assert float(np.abs(dphi * weight).max()) < 0.1


@pytest.mark.parametrize("case", ["rand1", "rand14", "real28", "real117", "real_speed125", "real_speed08", "rand128", "rand256",
                                  "rand510", *APP, *LONG])
def test_free_running_durations_match(weights, case):
    c = R.load_case(case)
    model = K.build(weights, explicit_noise=True, outputs=("pred_dur", "audio"))
    dur, audio = model(*inputs(c, forced=False))
    np.testing.assert_array_equal(dur.numpy(), c["pred_dur"])
    assert audio.shape == c["audio"].shape
    # Judge both against a float64 golden (same durations and noise): lokinfer
    # must be no worse than PyTorch f32 itself, within 3 dB.
    gold = R.golden_audio(c)
    ours, torch_f32 = R.snr_db(audio.numpy(), gold["audio"]), R.snr_db(c["audio"], gold["audio"])
    print(case, f"SNR vs f64 golden: lokinfer {ours:.1f} dB, torch f32 {torch_f32:.1f} dB")
    assert ours > torch_f32 - 3.0, (ours, torch_f32)


@pytest.mark.parametrize("case", ["real28", "app59f", "app59m", "s9f", "real117"])
def test_shipped_variant_generates_its_own_noise(weights, case):
    """The shipped model (ids, ref_s, speed, seed): noise generated in-kernel, like PyTorch after
    manual_seed. Same length as the reference, deterministic per seed, and as close to the reference
    audio as two PyTorch runs with different seeds are (research: correlation ~0.995)."""
    c = R.load_case(case)
    model = K.build(weights)
    ids = np.concatenate([[0], c["ids"], [0]]).astype(np.int64)
    a1 = model(ids, c["ref_s"], np.float32(c["speed"]), 1).numpy()
    a1b = model(ids, c["ref_s"], np.float32(c["speed"]), 1).numpy()
    a2 = model(ids, c["ref_s"], np.float32(c["speed"]), 2).numpy()
    assert a1.shape == c["audio"].shape
    np.testing.assert_array_equal(a1, a1b)  # deterministic per seed
    assert not np.array_equal(a1, a2)  # the seed matters
    corr = float(np.corrcoef(a1, c["audio"])[0, 1])
    print(case, f"shipped vs reference audio correlation {corr:.4f}")
    assert corr > 0.98, corr


@pytest.mark.parametrize("case", ["rand1", "real28", "real117", "rand510", *APP, *LONG])
def test_gpu_free_running_snr(weights, case):
    """M4 GPU path, free-running durations: same durations as the reference and audio SNR vs the f64 golden within
    3 dB of PyTorch f32 (a GPU-vs-CPU-graph comparison is not meaningful: both are valid f32 paths whose ulp-level
    differences grow through the LSTMs)."""
    c = R.load_case(case)
    model = K.build(weights, explicit_noise=True, outputs=("pred_dur", "audio"), gpu=True)
    dur, audio = model(*inputs(c, forced=False))
    np.testing.assert_array_equal(dur.numpy(), c["pred_dur"])
    gold = R.golden_audio(c)
    ours, torch_f32 = R.snr_db(audio.numpy(), gold["audio"]), R.snr_db(c["audio"], gold["audio"])
    print(case, f"GPU SNR vs f64 golden: {ours:.1f} dB (torch f32 {torch_f32:.1f} dB)")
    assert ours > torch_f32 - 3.0
