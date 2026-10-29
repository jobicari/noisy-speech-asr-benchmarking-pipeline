# v1: the first full run (`fullrun2000`)

v1 is the name for the first full benchmark run, labelled `fullrun2000` in the code, logs and GCS paths. It ran on 2026-10-14 and 2026-10-15. The next run, v2 (`fullrun2000b`), is planned in [`../v2_plan.md`](../v2_plan.md).

Details live in two files next to this one:

- [`results.md`](results.md): completeness checks, corpus WER per model and per corruption, the long-clip comparison.
- [`timing.md`](timing.md): the pieces of the run, wall time, cost, per-model GPU time, comparison with earlier sweeps.

Both were written on 2026-10-29 from the result pickles and sweep folders on Jo's Mac. This README summarises them; where a number here differs from those files, the files win.

## What was run

- 2000 LibriSpeech samples x 22 corruption types (including `none`) x 7 models = 308,000 transcriptions.
- Models: whisper-tiny, base, small, medium, large-v3, large-v3-turbo (Hugging Face backend), and wav2vec2-large-960h-lv60-self.
- Hardware: one NVIDIA A10 (24 GB) on Lambda at $1.29/h. `BATCH_SIZE=22`, `SAMPLES_PER_CALL=1`, `CHECKPOINT_SIZE=50`, 4 prep workers, thread caps 2, fp16.
- Decoding: greedy, Hugging Face defaults, no temperature fallback or compression-ratio check. Clips over 30 s used `WHISPER_LONGFORM=chunked`.
- Result file: `results/transcribe_cache/fullRun__hf__986ea79c__2000samples__fullrun2000.pkl` on the Mac. The GCS copy in `transcribe_cache/` is gone, so the Mac copy is the only one.

The run happened in pieces:

| Piece | What | Timing data |
|---|---|---|
| `fullrun2000` (attempt 1) | samples 1-316, crashed on the first clip over 30 s; 300 kept via checkpoints | yes |
| resume | samples 301-2000, with chunked long-form decoding | **no** (see gaps) |
| `fullrun2000_finish` | loaded the finished pickle from cache, no inference | phases only |
| `fullrun2000_finish_longform` | side run: the 8 clips over 30 s x 6 Whisper models, `WHISPER_LONGFORM=sequential` | yes |

## Results

Corpus WER % (all S+D+I over all reference words):

| Model | Clean | All 22 corruptions |
|---|---|---|
| whisper-tiny | 6.91 | 42.16 |
| whisper-base | 4.61 | 32.62 |
| whisper-small | 3.08 | 12.25 |
| whisper-medium | 2.56 | 7.90 |
| whisper-large-v3 | 1.76 | **5.16** |
| whisper-large-v3-turbo | 1.91 | 5.68 |
| wav2vec2-large-lv60-self | **1.59** | 20.00 |

- **whisper-large-v3 is the most robust model** (5.2% over all corruptions) and the best Whisper model on every corruption. Turbo follows at 5.7%.
- **wav2vec2 is best on clean audio** (1.6%) but fails under packet loss (`PL_5_.5`: 83% against 2.7% for large-v3), heavy clipping (`clip_025`: 34%) and white noise (`WN_3snr`: 40%).
- Clean numbers are in line with published LibriSpeech test-clean results (claim from `results.md`; no specific published figures are cited there).
- Two corruptions hurt even the large models: `BL_500_1000` (large-v3 33%) and the combined `WN_3-SV_.5-SDO_8_100_15` (large-v3 26%).
- Clips over 30 s (8 samples): sequential long-form decoding beats chunked for base and above, and is worse for tiny. Only 8 clips, so indicative.

The full per-corruption table is in [`results.md`](results.md).

### Repetition loops on `BL_500_1000`

tiny and base score over 300% WER on `BL_500_1000` (337% and 346%). Scoring is correct: the hypotheses really are repetition loops ("i am going to say that i am going to say that ...") that run the decoder to its 448-token limit. 90% of loop rows are on `BL_500_1000` and the combined corruption. The check is in [`../v2_readiness_review.md`](../v2_readiness_review.md).

Capping each row's errors at its reference length gives:

| Model | Loop rows | Corpus WER | Capped WER |
|---|---|---|---|
| tiny | 435 | 42.2% | 27.3% |
| base | 374 | 32.6% | 18.9% |
| small | 39 | 12.3% | 11.0% |
| medium | 7 | 7.9% | 7.7% |
| large-v3 | 1 | 5.2% | 5.1% |
| turbo | 6 | 5.7% | 5.4% |
| wav2vec2 | 0 | 20.0% | 20.0% |

On `BL_500_1000` alone, capping takes tiny from 337% to 98% and base from 346% to 96%. The ranking of the large models does not change.

## Timing and cost

- **About 4.0 h of container time and about $5.20** (instance setup and idle time not included).
- **About 21.8 inferences/s** over the whole main run (about 7.05 s and 154 inferences per sample, about 8.5 samples/min).
- **GPU-bound**: on attempt 1, GPU inference was 97.7% of wall time, the GPU was busy in 93% of nvidia-smi readings, and prep on 4 workers was fully hidden after the first sample.
- **large-v3 takes 30% of GPU time**; large-v3, turbo and medium together take 69%. wav2vec2 is 3%.
- **VRAM peak 21.8 of 23.0 GB** with all 7 models loaded at batch 22. There is little room for a larger batch or an eighth model.

The whole-run totals combine attempt 1's measured 2,222 s with the resume's ~12,000 s, which comes from the chat on 2026-10-15 and not from log files. They are not verified against logs.

## Known gaps

1. **Per-sample timing covers only the first 316 of 2000 samples.** The resume ran in a plain SSH shell that dropped, so its sweep folder, `timing.jsonl` and GPU/CPU logs were never saved. Per-model shares and rates in `timing.md` come from attempt 1 only, and `analyze.py --combine` cannot produce a total. Fixed for future runs on `main` (2591359: detached runs, `docker run -d`).
2. **Repetition loops** inflate tiny and base corpus WER (above).
3. **Tail percentiles over all 2000 samples are unknown** for the same reason as gap 1.
4. **46,200 rows have no `duration_s`** (the first 300 samples, which predate the field). Harmless: none of them is a long clip.
5. **`fullrun2000_finish`'s report is mostly dashes**: it only loaded the cached pickle. Expected, but it should not be read as a run.
6. **Earlier sweep folders are gone** (`worker_sweep_bs22`, `thread_caps`, `spc_sweep1`, `models7_*`); only `bs_sweep1` remains, on the Mac. v1 used a single config, so it cannot compare hyperparameters by itself.
7. **The result pickle has one copy**, on the Mac.
8. `analyze.py` quirks found while writing these reports (wrong `n_samples`, failed runs left out of `summary.csv`, inconsistent projection targets) and an out-of-date `bench/TIMING_SPEC.md` are listed in `timing.md`. The `n_samples` fix is on `run/full-rerun-readiness`.

## Removed claims

The original timing report cited a file `fullrun2000_first_look.md` for the result row counts and for a mean WER of 5.9 (turbo) vs 5.3 (large-v3). That file does not exist: it is not in the repository, its git history, Jo's checkout or elsewhere on the Mac, and Jo has never seen it. The mean WER figures had no source and were removed from `timing.md`. The turbo vs large-v3 comparison there now uses the corpus WER from `results.md` (5.68 vs 5.16), and the row count reference points to `results.md`, which checks the 308,000 rows directly.
