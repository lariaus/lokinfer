# Speech-to-text in lokinfer: design (2026-11-06)

Status: **researched and designed, not started**. The next step after the Qwen3-ASR phase-1 work; the user starts it
("you won't start working on it yet"). Research: [research-speech-to-text.md](research-speech-to-text.md). Todo:
docs/todo.md, "Speech-to-text in lokinfer".

## Goal (user, 2026-11-06)

**The full end-to-end path — audio in, text out — as a lokinfer binary and library, faster than the PyTorch / MLX
pipelines end to end**, with the same output as today's Python end-to-end path (up to computation differences).

## Decisions (user, 2026-11-06, three Q&A rounds)

| Topic | Decision |
|---|---|
| Where | Everything "audio in → text out" lives in lokinfer, at different layers (below); the app is a thin caller |
| Model | Only Qwen3-ASR-1.7B ships; the shared layers are generic (`tokenizer.json`, parameterized mel, generic loop) |
| Language | English only in the first version |
| Output | The same output as today's Python E2E path (english-practice: transformers, `language="English"`, greedy, `transcription_only`), up to computation differences; "do the right / correct thing" |
| E2E timing | **Warm**: model loaded, call → text (cold start / weight loading are not part of the goal) |
| Baselines | qwen-asr / transformers PyTorch **MPS** and **CPU**, **mlx-audio** (Python and Swift), **llama.cpp** mtmd |
| Devices | Mac first (binary + comparison); the iPhone gets the same library API later |
| Model files | `--model <name>`: the **C++ binary** downloads the pinned HF snapshot and converts it once into a cached lokinfer model folder; later runs reuse (mmap) it |
| Binary output | Text; options may print more (stats, JSON) |
| Long audio | Split only when the device can't hold the input in one pass; configurable |
| Input | float32 PCM + rate + channels; **lokinfer downmixes and resamples**; no codecs in lokinfer |
| Optimization | "Do the most optimal" to reach the goal (see "Performance design") |
| App switch | Wiring english-practice to it is a later, separate step |
| Modes | **Default = the reference behavior** (greedy; one pass unless the device can't hold it): every E2E test and baseline comparison uses it, exactly. Decoding and splitting are **configurable**, so later speed modes (deliberate splitting with parallel / batched passes, decoding beyond plain greedy, e.g. speculative) can be added; those are benchmarked as their own category, never mixed with the reference-matched rows (user, 2026-11-06) |

## Reference behavior to match (today's Python E2E path)

english-practice (`native-server/speech_to_text`):
1. AVFoundation decodes the file and converts it to 16 kHz mono float32 (AVAudioConverter).
2. `processor.apply_transcription_request(audio, language="English")` → features + prompt ids.
3. `model.generate(**inputs, max_new_tokens = int(128 + 8·seconds))`: greedy (`do_sample=false`), f16 on MPS.
4. `processor.decode(generated, return_format="transcription_only")`: special tokens skipped, the text after
   `<asr_text>` kept.
5. No splitting (one pass for the whole file), no peak normalization, no repetition fix.

lokinfer matches 2–4 exactly (features, prompt ids, greedy tokens, decoded text) and replaces 1 with its own downmix +
resampler (a computation difference: only the top mel bins can differ, research B). Splitting happens only beyond the
device's capacity (where the reference would run out of memory). Golden tests feed 16 kHz mono input, which bypasses
resampling, so they compare lokinfer with transformers exactly on the same samples.

## Layers

1. **Core** (unchanged): IR, compiler, runtime, kernels.
2. **Shared audio layer**: WAV reader (binary only), mean downmix, Kaiser-sinc polyphase resampler (torchaudio's closed
   form with the "kaiser_best" constants; 16 kHz input passes through), parameterized Whisper-style log-mel (n_mels,
   n_fft, hop, min length, pad multiple from the model's `preprocessor_config.json`), the quiet-point splitter.
3. **Shared text layer**: byte-level BPE tokenizer (own C++; vocab / merges converted from `tokenizer.json` into a
   compact binary when the model is prepared; NFC; a hand-coded Qwen2 pre-tokenizer), a streaming UTF-8-safe
   detokenizer, the generation loop (chunked prefill, decode steps over a session-owned KV cache, greedy argmax on the
   device, stops: eos set, max new tokens, KV capacity, cancel; per-token callback).
4. **Model package** `models/qwen3_asr`: the graphs + the recipe: feature parameters, the 1 s chunk layout, `N_a`, the
   hardcoded prompt ids (English), ≤ 2048-token prefill chunks with their audio slices, output parsing
   (`transcription_only`), `max_new_tokens = 128 + 8·seconds`, the capacity rule; one entry point.
5. **App**: capture / file decoding, UI, storage, options.

API sketch (C++, mirrored in Python):

```cpp
struct TranscribeOptions {
  // Defaults = the reference behavior (greedy, split only beyond capacity); other values are speed / quality modes,
  // benchmarked separately from the reference-matched rows.
  Decoding decoding = Decoding::Greedy;  // later: speculative, sampling (temperature / top-k / top-p, seeded)
  SplitPolicy split = SplitPolicy::WhenNeeded;  // later: Always (parallel / batched passes)
  double max_pass_seconds = 0;          // 0 = the device's capacity
  int max_new_tokens = 0;               // 0 = 128 + 8 * seconds (the reference rule)
  Device device = Device::Auto;
  std::function<bool(const Progress&)> on_progress;   // returns false to cancel
};
struct Transcription { std::string text; std::vector<Pass> passes; Timings timings; };
class SpeechToText {                    // loaded once (graphs, mmapped weights, tokenizer, mel tables), reusable
 public:
  static std::unique_ptr<SpeechToText> load(const std::string& model /* name or folder */, const LoadOptions&);
  Transcription transcribe(const float* pcm, size_t frames, int sample_rate, int channels /* interleaved */,
                           const TranscribeOptions&) const;
};
```

A lower-level path takes features directly (golden tests; like whisper.cpp's `whisper_set_mel`).

## Input contract

- Public: float32 PCM, interleaved, any rate (> 0) and channel count. lokinfer averages the channels, then resamples to
  16 kHz with the specified resampler; 16 kHz mono passes through unchanged.
- Canonical (model input): float32 mono 16 kHz [-1, 1], any length (the model pads < 0.5 s itself).
- No codecs: other formats are converted beforehand (`afconvert -f WAVE -d LEI16@16000 in.mp3 out.wav`, or
  `ffmpeg -i in -ar 16000 -ac 1 out.wav`).

## Long audio

- One pass whenever it fits: capacity = the device's KV cache / memory for `18 + N_a + max_new_tokens` positions (today
  ~20 min on the Mac, ~5 min on the iPhone).
- Otherwise qwen-asr's cut rule: near each boundary the quietest 100 ms window within ±5 s, cut at its quietest sample;
  no overlap; each pass gets its own features; texts concatenated (as qwen-asr: no separator).
- Configurable: max pass seconds, search window, quiet-window size.
- To check on the Mac: is > 20 min in one pass as accurate as split (qwen-asr's 1200 s is an unexplained constant)?

## The binary

```bash
$ utils/speech-to-text --input hello.wav --model qwen3-asr-1.7b
Hello, world.
$ utils/speech-to-text --input - --raw-f32 --rate 48000 --channels 2 --model qwen3-asr-1.7b < audio.f32
```

Flags: `--input FILE|-`, `--raw-f32 --rate --channels`, `--model NAME|DIR`, `--device cpu|gpu`, `--max-pass-seconds`,
`--max-new-tokens`, `--format txt|json` (json: text, passes, timings), `--timings` (stage breakdown to stderr). Text on
stdout, logs on stderr. The first use of a model name downloads + converts it into the cache
(`~/.cache/lokinfer/models/<name>`, pinned revision, progress on stderr).

## Performance design (warm E2E)

Model-only, lokinfer already leads (Mac GPU e2e 30 s: 1211 vs MLX 1288 ms), but only by ~5 %, and decode is ~80 % of
it. In order of expected impact:
1. **Argmax on the device**; the token id stays in a GPU buffer the next step's embedding reads; the host reads 4 bytes
   per token (no 594 KB logits copy, no host scan).
2. **Pipelined decode**: enqueue step N+1 before reading token N (MLX's trick, without Python), then K steps per commit
   with the ICB replay (≤ K−1 wasted steps at EOS).
3. **No per-call setup**: pipelines, memory plans and the KV cache ready at load; a new clip length (a new prefill
   shape) costs ms, not a re-plan / re-record.
4. **C++ front end**: resample + mel with vDSP (or on the GPU), overlapped with the encoder (features per 8 s window);
   tokenizer and detokenizer in C++ (µs per token).
5. **Long audio**: the next pass's encoder + prefill overlapped with the current pass's decode.
6. Measured with the `--timings` breakdown; the warm-call row is the goal, the first call reported separately.

## Tests

- Front end: the resampler vs `torchaudio.functional.resample` (kaiser_best params), ~1e-6; mel vs the HF extractor
  ≤ 1e-5 on 16 kHz input; the WAV reader on files from `afconvert` / `say` (FLLR chunk), EXTENSIBLE, 8 / 16 / 24 /
  32-bit and float.
- Text: tokenizer encode / decode vs the Rust `tokenizers` (fixed cases + fuzz), streamed decode = full decode, prompt
  ids = `apply_transcription_request` for several durations.
- End to end: generated ids = transformers `generate` (greedy) on the golden clips, a first difference only at a
  near-tie (the model-level rule); text = `decode(transcription_only)`.
- Speed: the E2E benchmark vs the five baselines (warm, 2 s … 30 min), the text equal to the reference required.

## E2E benchmark protocol

- Scope: a library call → text in a process with the model loaded (warm), the same audio file, English, greedy, the
  same `max_new_tokens`; the baselines' equivalent calls in a loaded process (qwen-asr / transformers `generate` on MPS
  and CPU, mlx-audio Python `generate`, the mlx-audio-swift Mac CLI, llama.cpp mtmd).
- Cases: 2 s, 5 s, 10 s, 30 s, 60 s, 5 min, 20 min, 30 min (16 kHz mono) + 48 kHz stereo variants of 5 s / 30 s.
- Protocol: decision 11 (≥ 2 warm-ups, ≥ 7–30 runs, median), the first call separately, the stage breakdown recorded;
  display asleep, big-model lock; estimate the sweep before running (~30–45 min for all impls: per impl, short first).

## Not covered (later)

Live streaming (partial transcripts while recording), timestamps (Qwen's separate ForcedAligner model), VAD, other
languages, cold-start optimization (prefetch, Metal binary archives), the iPhone binding, switching english-practice.
Kokoro's front end stays in english-practice (linguistic, language-specific); not revisited.
