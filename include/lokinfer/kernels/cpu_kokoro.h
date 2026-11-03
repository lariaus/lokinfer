// Typed CPU kernels for the Kokoro-82M port (kokoro-port-plan.md, "Basic CPU kernel set").
// All f32, contiguous row-major; `[C, L]` = channel-first. No exceptions; inputs are assumed valid
// (the Python type rules check shapes). Kernels that need temporaries (im2col tiles, LSTM gate
// projections, attention scores) use a grow-only thread-local scratch arena (kokoro_common.h): it
// allocates only the first time a larger size is seen, never per call in steady state.
#pragma once

#include <cstdint>

namespace lk::kernels::cpu {

// --- token-rate ([T, C] row-major) -------------------------------------------------------------

// out[t, :] = table[ids[t], :]
// Returns false (and writes nothing) if an id is outside [0, v).
bool embedding_f32(const float* table, const int64_t* ids, float* out, int64_t t, int64_t d, int64_t v);
// F.layer_norm over the last dim (two-pass stats).
void layernorm_f32(const float* x, const float* gamma, const float* beta, float* out, int64_t rows, int64_t cols,
                   float eps);
// (1 + gb[:C]) * LN_noaffine(x) + gb[C:]
void adalayernorm_f32(const float* x, const float* gb, float* out, int64_t rows, int64_t cols, float eps);
// Non-causal SDPA, no mask. q/k/v/out are [T, heads*headDim]; scale 1/sqrt(headDim).
void attention_f32(const float* q, const float* k, const float* v, float* out, int64_t t, int64_t heads,
                   int64_t headDim);
// 1-layer bidirectional LSTM (gate order i,f,g,o). w_ih [8H, I], w_hh [8H, H], b [8H] (= b_ih + b_hh);
// rows [0, 4H) forward, [4H, 8H) backward. out [T, 2H] = [h_fwd_t, h_bwd_t].
void lstm_bidir_f32(const float* x, const float* wih, const float* whh, const float* b, float* out, int64_t t,
                    int64_t in, int64_t hidden);
// Same, whhT = true: whh is w_hh^T [H, 8H] (the step GEMV runs on SME2 with the gates in ZA when available).
void lstmBidir(const float* x, const float* wih, const float* whh, bool whhT, const float* b, float* out, int64_t t,
               int64_t in, int64_t hidden);
// Attention over a fused QKV projection qkv [T, 3D] (biases added) -> out [T, D]; causal = GPT-2's mask.
void attention_qkv_f32(const float* qkv, float* out, int64_t t, int64_t heads, int64_t headDim, bool causal);
// dur[t] = max(1, rint(sum_j sigmoid(x[t, j]) / speed)), total[0] = sum dur.
void durations_f32(const float* x, float speed, int64_t* dur, int64_t* total, int64_t t, int64_t n);
// x.repeat_interleave(dur, 0) into out[frames, C] (rows past `frames` are dropped, missing rows zeroed).
// Returns false (and writes nothing) unless the durations are >= 0 and sum to `frames`.
bool repeat_interleave_f32(const float* x, const int64_t* dur, float* out, int64_t t, int64_t c, int64_t frames);
// out[r, :] = v
void broadcast_rows_f32(const float* v, float* out, int64_t rows, int64_t cols);
// torch.cat(dim=0) of n blocks [rows[i], cols]
void concat_rows_f32(const float* const* xs, const int64_t* rows, int64_t n, float* out, int64_t cols);
// torch.cat(dim=1) of n blocks [rows, cols[i]]
void concat_cols_f32(const float* const* xs, const int64_t* cols, int64_t n, float* out, int64_t rows);

// --- elementwise -------------------------------------------------------------------------------

void gelu_tanh_f32(const float* x, float* out, int64_t n);
void leaky_relu_f32(const float* x, float* out, int64_t n, float slope);
void scale_f32(const float* x, float* out, int64_t n, float s);
void add3_scale_f32(const float* a, const float* b, const float* c, float* out, int64_t n, float s);

// --- frame-rate ([C, L] channel-first) ---------------------------------------------------------

// F.conv1d (zero padding). w [co, ci, k], bias may be null. Lo = (l + 2p - d(k-1) - 1) / s + 1.
void conv1d_f32(const float* x, const float* w, const float* bias, float* out, int64_t ci, int64_t l, int64_t co,
                int64_t k, int64_t stride, int64_t padding, int64_t dilation);
// F.conv_transpose1d. w [ci, co, k], bias may be null. Lo = (l-1)s - 2p + k + op.
void conv_transpose1d_f32(const float* x, const float* w, const float* bias, float* out, int64_t ci, int64_t l,
                          int64_t co, int64_t k, int64_t stride, int64_t padding, int64_t outputPadding);
// Polyphase F.conv_transpose1d(leaky(x, slope)) + res with K = taps·s, 2p = (taps-1)·s, then an optional 1-sample
// left reflect pad before the residual add: wp from ops.convt_poly_weights_cpu, res/out [co, l·s + padLeft].
void conv_transpose1d_poly_f32(const float* x, const float* wp, const float* bias, const float* res, float* out,
                               int64_t ci, int64_t l, int64_t co, int64_t stride, int64_t padding, int64_t taps,
                               int64_t padLeft, float slope, const float* x2 = nullptr, const float* x3 = nullptr,
                               float inScale = 1.f);  // x2/x3: input = ((x + x2) + x3) * inScale
// Depthwise F.conv_transpose1d(groups=C). w [c, 1, k], bias may be null.
void conv_transpose1d_dw_f32(const float* x, const float* w, const float* bias, float* out, int64_t c, int64_t l,
                             int64_t k, int64_t stride, int64_t padding, int64_t outputPadding);
// (1 + gb[:C, None]) * instance_norm(x) + gb[C:, None] (biased var, two-pass stats).
void adain1d_f32(const float* x, const float* gb, float* out, int64_t c, int64_t l, float eps);
// x + (1/alpha) * sin(alpha * x)^2, alpha per channel.
void snake_f32(const float* x, const float* alpha, float* out, int64_t c, int64_t l);
// Nearest upsample by an integer factor: out [c, l*scale].
void upsample_nearest1d_f32(const float* x, float* out, int64_t c, int64_t l, int64_t scale);
// F.pad(mode='reflect'): out [c, left + l + right]; requires left, right < l.
void pad_reflect1d_f32(const float* x, float* out, int64_t c, int64_t l, int64_t left, int64_t right);

// Kokoro SourceModuleHnNSF(SineGen(24 kHz, x300, 8 harmonics, voiced threshold 10)) in torch's exact op
// order. f0 [n] (80 Hz), noise [300n, 9] (standard normal, the randn_like draw), w [9], b [1] -> out [300n].
void harmonic_source_f32(const float* f0, const float* noise, const float* w, const float* b, float* out, int64_t n);
// Same, with the SineGen noise generated in-kernel (Philox4x32-10 + Box-Muller) from `seed`, like
// torch's randn_like after torch.manual_seed: deterministic per seed, not torch's bit stream.
void harmonic_source_rng_f32(const float* f0, uint64_t seed, const float* w, const float* b, float* out, int64_t n);
// torch.stft(center, reflect pad, periodic hann(nFft), onesided) -> cat(abs, angle): out [nFft+2, s/hop + 1].
void stft_magphase_f32(const float* x, float* out, int64_t s, int64_t nFft, int64_t hop);
// torch.istft(exp(x[:nFft/2+1]) * e^{i sin(x[nFft/2+1:])}, periodic hann, center): x [nFft+2, l] -> out [hop(l-1)].
// Returns false (nothing written) when nFft / 2 + 1 > 64.
bool istft_logits_f32(const float* x, float* out, int64_t l, int64_t nFft, int64_t hop);

// --- fused frame-rate conv (kokoro_conv_fused.cpp) -------------------------------------------

enum class ConvAct { None, Snake, Leaky };

constexpr int64_t kStatChunk = 128;  // samples per (mean, M2) stats partial

// out = (conv1d(act((x - mean) * scale + beta), w) + bias [+ residual]) * outScale, stride 1, zero padding (applied
// after act). affine = f32[3, ci] rows (mean, scale, beta) (null = identity); alpha per input channel;
// stats (if set) receives per-(out channel, kStatChunk-sample chunk) (mean, M2) as f32[Co, ceil(Lo / 128), 2].
// Lo = l + 2·padding - dilation·(k-1); tile (time samples per task) must be a multiple of kStatChunk.
struct ConvFusedArgs {
  const float* x = nullptr;
  const float* w = nullptr;  // [co, ci, k]
  const float* bias = nullptr;
  const float* affine = nullptr;  // [3, ci]: mean, scale, beta
  const float* alpha = nullptr;  // snake
  const float* residual = nullptr;  // [co, Lo]
  float* out = nullptr;
  float* stats = nullptr;
  int64_t ci = 0, l = 0, co = 0, k = 0, dilation = 1, padding = 0, tile = 768;
  ConvAct act = ConvAct::None;
  float slope = 0.f;  // leaky
  float outScale = 1.f;  // out = (conv + bias + residual) * outScale (AdainResBlk1d's 1/√2)
  const float* x2 = nullptr;  // optional 3-input mean prologue: the input is ((x + x2) + x3) * inScale (Winograd only)
  const float* x3 = nullptr;
  float inScale = 1.f;
};
void conv1d_fused_f32(const ConvFusedArgs& p);
// The same conv as a Winograd conv (F(4,3) for K = 3, nested F(4,4) with `groups` tap groups for K = 4..12):
// up = ops.wino_weights_cpu's U [A][ceil(co/32)][ci·groups][32], tr its [BT | AT] table. GEMMs on SME2 when the CPU
// has it, else Accelerate. Short sequences (fewer than 2 tiles per pool thread) run as one tile in two parallel phases.
void conv1d_wino_f32(const ConvFusedArgs& p, const float* up, const float* tr, int64_t groups);
int64_t statChunks(int64_t l);  // ceil(l / kStatChunk)
// Per-(channel, chunk) (mean, M2) of x [c, l] -> stats [c, ceil(l / 128), 2].
void channel_stats_f32(const float* x, float* stats, int64_t c, int64_t l);
// AdaIN as affine [3, c] = (mean, (1 + gb[:c]) / sqrt(var + eps), gb[c:]) from tile stats over length l
// (exact two-pass combine in double, biased variance).
void adain_affine_f32(const float* stats, const float* gb, float* affine, int64_t c, int64_t l, float eps);

}  // namespace lk::kernels::cpu
