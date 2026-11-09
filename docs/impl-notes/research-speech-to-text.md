# Research: end-to-end speech-to-text (Qwen3-ASR-1.7B), 2026-11-06

Four parallel research passes for the speech-to-text design ([speech-to-text-design.md](speech-to-text-design.md)):
(A) how other stacks package audio-in → text-out, (B) the audio front end, (C) the shared text layer, (D) end-to-end
speed. Primary sources (GitHub raw files, docs) as of 2026-11-06; items marked [measured] were run locally against the
pinned snapshot `Qwen/Qwen3-ASR-1.7B-hf@bcd2b5b7` and transformers 5.20.2.

## A. How other stacks package it

| Stack | Library entry point | Input | Who converts (rate / channels) | Codecs in lib | Long audio |
|---|---|---|---|---|---|
| whisper.cpp | `whisper_full(ctx, params, const float* pcm, n)` | f32 mono, 16 kHz implied | caller (CLI: miniaudio) | no | 30 s windows, timestamp seek; Silero VAD inside |
| llama.cpp libmtmd | `mtmd_bitmap_init_from_audio(n, const float*)` | f32 mono at the model rate | caller (helper: miniaudio) | helper only | Qwen3: 800-frame (8 s) encoder windows, whole audio in one LLM context |
| sherpa-onnx | `AcceptWaveformOffline(stream, sample_rate, samples, n)` | f32 mono [-1, 1] + rate | **library** (Kaldi windowed sinc) | WAV (mono s16) only | none inside; app-side VAD segments |
| qwen-asr (official) | `transcribe(audio, context, language)` | path / URL / base64 / `(ndarray, sr)` | **library** (mean downmix, librosa soxr_hq, peak-normalize if > 1) | yes | quiet-point split at 1200 s, texts joined with `""` |
| mlx-audio (Py) | `load(repo).generate(audio, language=...)` | path, or array (16 kHz assumed) | lib for files (miniaudio / ffmpeg + kaiser_best) | yes | same 1200 s split |
| mlx-audio-swift | `generate(audio: MLXArray, ...)` | MLXArray at `sampleRate` | caller (AVAudioConverter) | via AVAudioFile | same 1200 s split |
| HF pipeline | `pipe(inputs, chunk_length_s=...)` | ndarray (rate assumed) or `{raw, sampling_rate}` | pipeline (torchaudio) | ffmpeg subprocess | overlapping chunks + LCS merge |

Lessons:
- **Everyone's model input is f32 mono 16 kHz [-1, 1]**; stacks only differ on who converts. Implied-rate APIs get
  misused (mlx-audio-swift's own Qwen3-ASR README example feeds a 44.1 kHz file unresampled). A `(pcm, rate,
  channels)` signature with explicit interleaving avoids the bug class (HF had to add an error for ambiguous 2-D
  shapes).
- **No core library carries codecs**; CLIs bundle miniaudio or shell to ffmpeg, and whisper.cpp's README tells users
  to convert with ffmpeg first.
- **Load once, run many** (whisper ctx/state, sherpa recognizer/stream): a model object loaded once, a per-call
  transcription.
- **The model-specific feature recipe lives with the model** (mtmd: a preprocessor subclass per projector type,
  parameterized from the model file).
- llama.cpp and sherpa-onnx already run Qwen3-ASR (GGUF; three ONNX graphs). mtmd's `_qwen3a` mel extractor is a
  usable C++ reference for the features.
- qwen-asr's transformers backend defaults `max_new_tokens = 512` (vLLM 4096, mlx-audio 8192): a long pass can be
  silently truncated; size the limit from the audio length.

## B. Audio front end (exact)

Features (`Qwen3ASRFeatureExtractor`, = Whisper's log-mel with Qwen's length handling) [measured: a 15-line numpy f64
port matches HF to ≤ 8.3e-6]:
1. Input f32 mono 16 kHz; no gain, DC removal, pre-emphasis or dither.
2. n < 8000 samples (0.5 s): zero-pad to 8000; the padding counts as valid frames.
3. STFT: n_fft 400, hop 160, **periodic** Hann, `center=True` with **reflect** padding (200 each side), onesided
   (201 bins); the last frame is dropped → F = n // 160 frames.
4. Power |X|²; 128 mel filters, 0–8 kHz, **Slaney** mel scale + Slaney area normalization (= librosa defaults =
   Whisper's `mel_128`).
5. `L = log10(max(mel, 1e-10))`; `L = max(L, max(L) − 8)` with **the max over the whole pass**; `L = (L + 4) / 4`.
   Features are therefore not streamable: a pass is extracted whole before chunking.
6. Pad the time axis to a multiple of 100 frames with **0.0** (after normalization; the encoder's last partial
   chunk sees it, so it is part of the contract) → `[n_chunks, 128, 100]`.
7. Audio tokens: `N_a = 13·(F // 100) + (r ? c(c(c(r))) : 0)`, r = F % 100, `c(x) = (x − 1)//2 + 1`.
Differences from Whisper: no 30 s pad/trim, the 0.5 s minimum, the 0.0 pad to whole seconds, the token formula.

Resampling — what references use and what it changes:

| Stack | Resampler |
|---|---|
| qwen-asr, librosa | soxr HQ (libsoxr, multi-stage, not closed form) |
| HF pipeline, vLLM | torchaudio `resample` defaults (Hann sinc, width 6, rolloff 0.99) |
| openai-whisper, mlx-whisper, faster-whisper | ffmpeg swr (Kaiser sinc, 1024 phases) |
| whisper.cpp, mtmd | miniaudio linear + 4th-order low-pass |
| english-practice today | AVAudioConverter (AVFoundation) |

Experiment (48 / 44.1 kHz synthetic speech and noise through each resampler and the HF extractor): every good
band-limited resampler agrees to ≤ 1e-3 feature units below mel bin 120 (~6.4 kHz); differences show only in the top 8
bins (up to 0.7 units = 28 dB in quiet cells); torchaudio's default aliases 8.5 kHz content at only −11 dB. **Recommended:
Kaiser-windowed sinc polyphase in torchaudio's closed form with the "kaiser_best" constants** (width 64, rolloff
0.9475937167399596, β 14.769656459379492): vs soxr HQ identical to < 1e-4 below bin 120, ≤ 0.12 units above.
Reference vectors from one `torchaudio.functional.resample(..., resampling_method="sinc_interp_kaiser", ...)` call.
Table sizes: 48k 1×409 taps, 44.1k 160×815 (0.5 MB f32), 8k 2×137; rates with a tiny gcd with 16000 need taps computed
on the fly or rejection. 16 kHz input bypasses the resampler (bit-exact).

Downmix: **arithmetic mean of channels** (qwen-asr, librosa, HF, vLLM, miniaudio); ffmpeg differs (0.707·(L+R) for float
output). Before resampling.

WAV parser scope: RIFF/WAVE, walk chunks (skip unknown, honor the odd-size pad byte); `fmt ` incl. WAVE_FORMAT_EXTENSIBLE
(sub-format GUID); PCM 8 (unsigned) / 16 / 24 (packed) / 32 bit and float 32 / 64, scaled by 2^(bits−1); `data` size 0 or
0xFFFFFFFF = to end of file, clamp oversized; reject μ-law / A-law / ADPCM / RIFX / AIFF with a "convert first" message.
**`afconvert` and `say` insert a 4 KB `FLLR` chunk before `data`** [measured]: a fixed-offset parser breaks on them.

Long audio: Whisper's sequential seek (needs timestamp tokens) and HF overlap + LCS merge don't fit Qwen3-ASR; VAD needs
another model. qwen-asr's rule (quietest 100 ms window within ±5 s of each boundary, cut at its quietest sample, no
overlap, concatenate; pieces < 0.5 s zero-padded) is deterministic and testable against `reference.split_long`. 1200 s
is only a code constant in qwen-asr (no validated limit published). Each pass has its own features (global max).

## C. Shared text layer

Qwen3-ASR's tokenizer [measured]: byte-level BPE, 151,643 vocab entries + 151,387 merges + 62 added tokens (ids
151643–151704); **NFC normalizer**; pre-tokenizer regex GPT-4 style but single digits, with a negative lookahead
`\s+(?!\S)`; ByteLevel decoder (pure byte concatenation); `vocab_size` 151,936 → logits rows ≥ 151,705 have no token.
`<asr_text>` (151704) is **not** special: it survives `skip_special_tokens`.

Options for a C++ tokenizer:
- HF `tokenizers` (Rust) via FFI (mlc-ai/tokenizers-cpp): exact by construction, but a Rust toolchain + per-arch cargo
  builds + a multi-MB library, and no streaming decode.
- llama.cpp's own BPE (`llama-vocab.cpp`): hand-coded Qwen2 splitter (`unicode_regex_split_custom_qwen2`, added after
  `std::regex` overflowed the stack, #21257); MIT; tied to GGUF structures → borrow the algorithm, not the code.
- sentencepiece: wrong family. Regex-engine ports (PCRE2 / std::regex) failed on a Qwen-family tokenizer
  (executorch#14432).
- **Recommended**: own C++17 byte-level BPE (~1k lines + Unicode tables), loading a compact binary that a Python step
  converts from `tokenizer.json` when the model is prepared; tested token-for-token against the Rust `tokenizers`
  (fixed cases from llama.cpp's tokenizer tests + fuzzing).

Streaming detokenization: Qwen2's decoder is pure byte concatenation, so "accumulate bytes, emit up to the last complete
UTF-8 character" (llama.cpp `validate_utf8`) equals HF's full decode, O(1) per token (e.g. 🎉 = 3 tokens).

Prompt: hardcode it in `models/qwen3_asr` (a Jinja engine buys nothing for one fixed-format prompt; the template even
uses the HF-only `{% generation %}` tag). Exact ids for English [measured]:
`151644 8948 198 151645 198 | 151644 872 198 151669 151676×N_a 151670 151645 198 | 151644 77091 198 11528 6364 151704`
(18 + N_a ids; without a language the last three are dropped and the model writes `language X<asr_text>` itself).

Generation loop: only the last row through the LM head on prefill chunks (llama.cpp output-row selection, mlx-lm);
**argmax on the device** (Qwen3-ASR is greedy: one int read back instead of 594 KB of logits per step); then keep the
token id on the device and enqueue step N+1 before reading token N (mlx-lm `async_eval`). Stops: eos {151643, 151645},
max new tokens, KV capacity, cancel. KV cache: 112 KiB per token (f16), caller/session owned.

## D. End-to-end speed

Model-only today (M2 Max, warm, median): lokinfer GPU e2e 320 ms (5 s) / 1211 ms (30 s) vs the best competitor (MLX)
337 / 1288 — a ~5 % lead. Decode is ~77–80 % of it (30 s: 80 tokens × ~11.7 ms).

Outside the model (warm process, model loaded — the user's E2E definition):
- **qwen-asr / HF on MPS**: `generate` syncs the host on every token (torch MPS decode 24 ms/token vs lokinfer 11.7);
  full f32 logits row each step; features on the CPU via torch (~0.09 ms per audio second [measured]: 30 s 2.5 ms,
  20 min 110 ms); tokenizer decode in Python.
- **mlx-audio (Py / Swift)**: argmax on the device and `async_eval` pipelining (the next step is enqueued before the
  host reads the token); features via transformers on the CPU (Py) or `rfft` on the GPU (Swift).
- **llama.cpp**: C++ loop, but copies the full f32 logits to the host each token and samples on the CPU; mel on the
  CPU (4 threads).
- Cold costs (not in the goal, for reference): Python imports 1–3 s [measured], torch.compile first call ~100 s, MLX
  reads/copies the 3.8 GB of weights (no zero-copy mmap); llama.cpp mmaps (no copy) but compiles Metal source at init
  in its default build.

Where lokinfer can win warm E2E: on-device argmax + token feedback + pipelined (or K-steps-per-commit replayed) decode;
LM head only where needed; C++ front end (resample + mel, vDSP or GPU) overlapped with the encoder; no per-call
allocation / pipeline creation (first call per new prefill shape must be cheap); the greedy loop and detokenization in
C++. Measurement protocol proposed: warm-call (model loaded, ≥ 2 warm-ups, dashboard protocol) as the primary row, with
first-call reported separately; clips 2 s … 30 min plus a 48 kHz stereo variant; per-stage `--timings`; the same text as
the reference required.
