# v2 (fullrun2000b): timing and throughput

Source: the `[TIMING]` records, `run.json`, `gpu.csv` and `cpu.csv` in `results/lambda_bench_runs/fullrun2000b/`, `fullrun2000b_longform/` and `fullrun2000b_loops/` on Jo's Mac, plus v1's attempt 1 in `fullrun2000/`. Every number comes from [`timing.py`](timing.py). Computed 2026-10-30.

Config: one NVIDIA A10 (22 GiB usable) on Lambda at $1.29/h, 30 host CPUs, `BATCH_SIZE=22`, `SAMPLES_PER_CALL=1`, `CHECKPOINT_SIZE=50`, 4 prep workers, thread caps 2, fp16, HF backend. Image and checkout are both `main` at `5fe7566`, transformers 5.19.0, torch 2.14.1+cu126.

## 1. Timing data is complete

Unlike v1, the whole run has timing:
- 2,000 `prep` records and 14,000 `sample` records, one per sample per model, all 2,000 samples
- all 40 `checkpoint` records
- `startup`, 7 `model_load`, `finalize` and the `run_config` record

The run happened in one piece, with no crash and no resume.

## 2. The whole run

| Piece | When (UTC, 2026-10-29) | Wall | Cost | Inferences | Throughput |
|---|---|---|---|---|---|
| Main run `fullrun2000b` | 04:56-09:19 | 15,770 s (4.38 h) | $5.65 | 308,000 | 19.65 inf/s steady |
| Long-clip pass `_longform` | 09:19-09:24 | 255 s | $0.09 | 1,056 | 4.6 inf/s (30-35 s clips) |
| Looped-clip pass `_loops` | 09:24-10:11 | 2,848 s (0.79 h) | $1.02 | 6,462 | 2.3 inf/s (fallback decoding) |
| Finders, analysis, uploads, results copies, terminate | to about 10:15 | about 4 min | about $0.10 | | |
| **Total** | **04:56 to about 10:15** | **about 5.3 h** | **about $6.85** | 315,518 | |

The instance's own setup time (build, smoke tests) before 04:56 is not included. The end time is an estimate: the uploaded copy of the wrapper log is taken just before the upload, so the upload and terminate lines aren't in it.

Main run unit costs:
- **$0.018 per 1,000 transcriptions**
- **$0.0086 per audio hour per model**: 93.8 h of corrupted audio through 7 models
- **RTF 0.0066 per model**: one model transcribes one hour of audio in 24 s

## 3. Where the main run's time went

| Part | Seconds | Share of wall |
|---|---|---|
| Startup (dataset, IR, pool) | 11.3 | 0.1% |
| Model loads (7, from the HF cache) | 10.8 | 0.1% |
| **GPU inference** | **15,507.9** | **98.3%** |
| Scoring (WER/CER) | 92.6 | 0.6% |
| Waiting for prep workers | 7.1 | 0.0% |
| Fetching from the dataset stream | 2.3 | 0.0% |
| Checkpoint saves (40 to GCS) | 59.2 | 0.4% |
| Finalize (merge 40 checkpoints, save, clean up) | 74.2 | 0.5% |
| Unaccounted (container start, imports, exit) | 5.1 | 0.0% |

- **The run is entirely GPU-bound.** Everything except inference adds up to under 4 minutes over 4.4 hours.
- **Prep is fully hidden.** The 4 workers spent 3,351 CPU-seconds corrupting audio (p50 1.34 s per sample, max 11.3 s), and the main process waited 7 s in total. More workers or fewer threads wouldn't speed this run up.
- **The rate was steady throughout.** A checkpoint (50 samples) took 396 s at the median (325-468 s), 393 s on average in the first half and 392 s in the second.
- **GPU**:
  - mean utilisation 67% (p50 78%), busy in 89% of the half-second readings
  - mean power 132 W, max 170 W
  - **VRAM peaked at 22,333 MiB, 256 MiB below the card's 22.06 GiB**

## 4. Per model

| Model | GPU time (s) | Share | Inf/s | Audio s/s | Per-sample p50 (s) | p90 (s) | p99 (s) | Max (s) |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 1,724 | 11.1% | 25.5 | 196 | 0.36 | 2.30 | 2.60 | 3.80 |
| whisper-base | 1,756 | 11.3% | 25.1 | 192 | 0.41 | 2.92 | 3.36 | 4.27 |
| whisper-small | 1,582 | 10.2% | 27.8 | 213 | 0.65 | 1.01 | 5.05 | 6.89 |
| whisper-medium | 2,796 | 18.0% | 15.7 | 121 | 1.27 | 1.84 | 2.67 | 10.16 |
| whisper-large-v3 | 4,354 | 28.1% | 10.1 | 78 | 2.04 | 2.77 | 3.76 | 14.42 |
| whisper-large-v3-turbo | 2,886 | 18.6% | 15.2 | 117 | 1.41 | 1.57 | 1.78 | 3.91 |
| wav2vec2-large | 409 | 2.6% | 107.6 | 826 | 0.15 | 0.40 | 0.79 | 1.25 |

(A "sample" is one call with all 22 corrupted clips of one LibriSpeech sample.)

- **large-v3 alone is 28% of the GPU time.** large-v3, turbo and medium together are 65%. Dropping or replacing one of them is the only lever that moves total time much.
- **Turbo is 1.5x faster than large-v3** for 0.5 points of WER over all corruptions (5.68 against 5.16).
- **tiny, base and small cost the same** (about 1,600-1,750 s each), and tiny is slower than small. Their medians scale with size (0.36, 0.41, 0.65 s), but their tails don't. See the next section.
- **wav2vec2 (CTC, no decoder loop) is 10.6x faster than large-v3** and 2.6% of the run.

## 5. What repetition loops cost

Sample calls in which at least one of the 22 clips looped, against calls with no loop:

| Model | Calls with a loop | Mean s (loop) | Mean s (no loop) | GPU time in loop calls |
|---|---|---|---|---|
| tiny | 624 | 2.02 | 0.34 | 1,259 s, **73%** of tiny's time |
| base | 478 | 2.41 | 0.40 | 1,152 s, **66%** |
| small | 46 | 4.72 | 0.70 | 217 s, 14% |
| medium | 6 | 9.78 | 1.37 | 59 s, 2% |
| large-v3 | 1 | 14.42 | 2.17 | 14 s |
| turbo | 7 | 3.41 | 1.44 | 24 s |

- **A loop makes a call about 6-7x slower** (2.4x for turbo): the whole batch waits for its looping clip to reach the 448-token limit.
- **Loops cost 2,271 s, 14.6% of all inference**, measured against each model's loop-free mean. They are the whole reason tiny and base are as slow as small. Without loops, tiny would take about 680 s, not 1,724 s.
- The looped-clip pass (fallback) would cut this cost only if it replaced greedy decoding in the main run. Run as a separate pass, it adds 0.8 h.

## 6. Side passes

**Long-clip pass**: 8 samples x 22 corruptions x 6 Whisper models, sequential long-form decoding, 255 s.
- Inference is 80% of its wall time. Startup, model loads and waiting for prep weigh more in such a short run, because the 30-35 s clips take about 9 s each to prep.
- Per call, large-v3 takes 7.1 s (p50) and turbo 3.9 s.

**Looped-clip pass**: 1,077 clips in 959 samples (1.1 clips per sample on average) x 6 Whisper models with temperature fallback, 2,848 s.
- **It is slow per clip** (2.3 inf/s against 19.7 in the main run) for two reasons:
  - the batches are small (`SAMPLES_PER_CALL=20` gives about 22 clips per call, but from 20 samples)
  - every clip that loops is decoded up to 6 times at rising temperatures
- tiny and base take 57% of the pass; turbo only 5%.
- GPU utilisation is low (mean 37%), and VRAM peaked at 21,235 MiB.

## 7. v2 against v1

v2 is **10% slower than v1**: 19.65 inf/s steady against 21.99 for v1's attempt 1. Over the 316 samples both runs timed, sample by sample:

| Model | v1 (s) | v2 (s) | v2 / v1 |
|---|---|---|---|
| tiny | 189.8 | 233.5 | **1.23** |
| base | 219.4 | 265.3 | **1.21** |
| small | 206.6 | 245.3 | **1.19** |
| medium | 395.2 | 436.6 | 1.10 |
| large-v3 | 660.6 | 689.5 | 1.04 |
| turbo | 434.0 | 459.1 | 1.06 |
| wav2vec2 | 66.0 | 67.0 | 1.02 |
| **All** | **2,171** | **2,396** | **1.10** |

Per sample and model, the v2/v1 ratio is 1.09 at the median (p10 1.01, p90 1.31).

What is known:
- The transcripts are identical (see the results doc), so v2 did exactly the same decoding work: same tokens, same loops.
- The slowdown isn't the data mix: these are the same 316 samples. On the rest of its run, v2 was no slower (19.8 inf/s, against 20.3 on the first 316).
- It isn't prep, scoring or the pipeline: those stayed under 1% of wall time in both runs. The waiting time is 6-7 s in both.
- **Host CPU speed is about the same**: worker prep took 1.34 s p50 per sample in v2 and 1.31 s in v1. Both hosts have 30 CPUs. Container CPU use is similar (145% against 153% mean).
- GPU utilisation was lower in v2 (67% against 76%), and power slightly lower (132 W against 137 W).

**Interpretation (not proven):** the slowdown is largest for the small models (19-23%) and smallest for the big ones (4-6%). Small Whisper models spend most of a decoding step in per-token CPU-side work (Python generation loop, kernel launches) rather than GPU compute. That, together with the lower GPU utilisation, points at **more CPU-side overhead per token** in v2, not a slower GPU. Two candidates the data can't separate:
1. **Library versions.** v2's pins are rebuilt from PyPI release dates, assuming v1's image installed the newest packages on 14-15 Oct. If v1's image reused a Docker layer cached from an earlier build, it ran older versions, with a possibly lighter generation loop. v1 never logged its versions. v2 now does, in `run_config`.
2. **The instance.** A different physical host or driver with slower kernel launch latency. Same CPU count and similar prep speed make a weaker CPU unlikely, but launch latency isn't measured.

A short benchmark on one instance with both images (same 20 samples, tiny only) would settle it in about 10 minutes, if it matters.

Other v1 vs v2 differences:
- **VRAM peak 22,333 MiB in v2 against 21,809 in v1**, for identical work. v2 has no headroom left.
- Shares of GPU time per model are within 2.4 points of v1's.
- **Cost per audio hour per model: $0.0086 against $0.0074.**
