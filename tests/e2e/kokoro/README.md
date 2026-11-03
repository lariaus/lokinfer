# Kokoro-82M end-to-end quality test (english-practice → lokinfer)

Validates lokinfer's shipped Kokoro GPU model (`models/kokoro/model.py` `build(weights, gpu=True)`) against
english-practice's own TTS path, on 225 real and written texts, on the Mac GPU and the iPhone GPU — step 1 of the
app's "transition to new TTS" plan. Not part of pytest (nothing here is collected): it's a manual, run-once-per-
release check that produces a listening page.

| file | what |
|---|---|
| `texts.json` | the corpus (committed): id, text, expected length class, source, voice |
| `make_corpus.py` | regenerates `texts.json` (samples english-practice's data read-only, checkout located by `english_practice_dir` / `LOKINFER_ENGLISH_PRACTICE`, see docs/benchmarking.md § Local configuration; + the texts written here) |
| `common.py` | paths, run dirs, the app's WAV encoding, the app's similarity measure (copied from compare_packs.py) |
| `kokoro_model.py` | **verbatim copy** of english-practice `native-server/kokoro_tts/kokoro_model.py` (the reference) |
| `prepare.py` | step 1: per-chunk model inputs exactly as the app computes them |
| `run_backend.py` | step 2: synthesizes every text with one backend (`python`, `python_b`, `mac_gpu`, `ios_gpu`) |
| `compare.py` | step 3: the difference test, PASS / FAIL, `summary.json` |
| `page.py` | step 4: the listening page `index.html` |

Generated output goes to `e2e-runs/kokoro/<run>/` (git-ignored): `inputs.json/.npz`, `audio/<backend>/<id>.wav`,
`results/<backend>.json` (per text: samples, per-chunk samples, time), `kokoro_gpu.lkm` (the compiled model the
iPhone sessions upload), `rows.json`, `summary.json`, `index.html`.

## Corpus

225 texts: 22 single words, 18 short phrases, 92 sentences (the app's Auto Shadow TTS pack, shadowing DBs, phoneme
sentences, Questions themes), 36 long 5-10 s sentences (incl. written spoken-style passages), 6 written general questions, 8 paragraphs of
20-40 s, 14 long paragraphs of 40-110 s (2-5 chunks, written multi-paragraph expository texts), 35 edge cases (numbers,
currencies, dates, times, abbreviations, all caps, quotes, ellipses, dashes, parentheses, accented words, emoji,
e-mail / URL, line breaks, one-letter texts). Voices: ~70 % the app's "Kokoro random" pair (af_heart, am_echo), the
rest af_bella, af_nicole, af_sarah, am_michael, am_fenrir, bf_emma, bm_george, bf_isabella. Personal and third-party
app texts are deliberately excluded (the corpus is public). 263 chunks, about 1842 s of audio.

## How the app synthesizes (what `prepare.py` + the `python` backend reproduce)

`kokoro_frontend::Frontend::prepare(text, voicePack)` (a bit-exact C++ port of kokoro's `KPipeline`) gives the chunks:
`KPipeline` with lang_code `'a'` for every voice (British voices too), lines split on `\n+`, misaki G2P, `en_tokenize`
chunking, empty chunks skipped, phonemes truncated to 510; per chunk `inputIds` = vocab ids **without** the two 0 pads
and `refS` = voice pack row `len(phonemes) - 1`, selected from the length **before** the vocab filter. Then
`kokoro_tts.cpp` calls `kokoro_model.infer(inputIds, refS bytes, 1, seed)` per chunk (speed 1) and concatenates the
chunk audio into one 24 kHz 16-bit WAV. Here `prepare.py` runs the Python `KPipeline(model=False)` itself (which the C++
port matches bit for bit) and the `python` backend calls the copied `kokoro_model.py` exactly that way. The lokinfer
backends get the same chunks: `[0, *inputIds, 0]`, `refS`, speed 1.0, a seed.

## Running

From the repo root, with the repo `.venv` (`HF_HUB_OFFLINE=1` is set by the scripts). Locks: docs/benchmarking.md.

```sh
cd tests/e2e/kokoro
RUN=2026-11-03
../../../.venv/bin/python prepare.py --run $RUN
# Mac backends, under the Mac bench lock (timing)
for b in python python_b mac_gpu; do
  ../../../scripts/lk-lock /tmp/lokinfer-macbench.lock ../../../.venv/bin/python run_backend.py $b --run $RUN
done
# iPhone (LokiHost running as the remote executor: scripts/ios_deploy.sh), one phone-lock session per part
for p in 1 2 3; do
  ../../../scripts/lk-lock /tmp/lokinfer-phone.lock sh -c "../../../.venv/bin/python run_backend.py ios_gpu --run $RUN --part $p/3"
done
../../../.venv/bin/python compare.py --run $RUN
../../../.venv/bin/python page.py --run $RUN
cd ../../../e2e-runs/kokoro/$RUN && ../../../.venv/bin/python -m http.server 8766 --bind 127.0.0.1
# then open http://127.0.0.1:8766/
```

`run_backend.py` skips texts already done in that run (`--redo` reruns them), so an interrupted backend resumes;
`--ids=t140,t141,...` runs (and redoes) only those texts, e.g. after editing `texts.json` and re-running `prepare.py`.
The english-practice checkout and the iPhone UDID come from `lokinfer.local.json` (`english_practice_dir`,
`iphone_udid`) or `$LOKINFER_ENGLISH_PRACTICE` / `$LOKINFER_IPHONE`.
Seeds: chunk k of text i uses `SEED_BASE[backend] + 64 i + k`; `python` and both lokinfer backends share the seed
base (lokinfer's in-kernel Philox noise is not PyTorch's bit stream anyway), `python_b` uses another one.

## The difference test (`compare.py`)

Measure: english-practice's own `spectral_similarity` (native-server/tests/kokoro_port/fixtures/compare_packs.py):
cosine of the log-magnitude spectrograms (n_fft 1024, hop 256) of the two WAVs. Kokoro's decoder draws random noise
on every call, so waveforms are never bit-identical — not even PyTorch vs itself — while durations are deterministic.
The baseline is Python vs a second Python run with other seeds, on **every** text.

PASS when all of:
- every text has audio on every backend, all samples finite, no errors;
- every Mac GPU and iOS GPU duration (sample count of the text and of each chunk) equals Python's exactly;
- lokinfer↔Python median similarity ≥ baseline median − 0.001, and min ≥ baseline min − 0.005 (tolerances fixed
  before the first run; the baseline's own run-to-run spread on the same texts is larger than both).

Outliers (listed, not a FAIL by themselves): texts whose lokinfer↔Python similarity is more than 0.01 below the same
text's Python↔Python similarity, or below the baseline's 1st percentile.

Timing: model calls only (setup and a 6-call warm-up excluded), summed over a text's chunks. Python = PyTorch CPU in
process; Mac GPU = in process; iOS GPU = the phone's own run time (`dev.run`'s times; `wall_ms` adds the cable
transfer of the audio).
