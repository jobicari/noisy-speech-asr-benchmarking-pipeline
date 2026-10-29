# v1 (fullrun2000): timing and throughput

Source: `results/lambda_bench_runs/fullrun2000/`, `fullrun2000_finish/` and `fullrun2000_finish_longform/` on Jo's Mac (same three folders as in GCS), read with the `[TIMING]` records described in `bench/TIMING_SPEC.md`. Computed 2026-10-29. Config for all of it: A10 24 GB, $1.29/h, `BATCH_SIZE=22`, `SAMPLES_PER_CALL=1`, `CHECKPOINT_SIZE=50`, 4 prep workers, thread caps 2, HF backend, fp16.

## Data check
- **Results are complete**: 308,000 rows in `fullRun__hf__986ea79c__2000samples__fullrun2000.pkl` (see `results.md`), plus 1,056 rows in the long-clip side run.
- **Timing is not complete.** The full run happened in four pieces, and only two have per-sample timing:

| Piece | When (UTC) | What | Wall | Timing data | Cost |
|---|---|---|---|---|---|
| `fullrun2000` (attempt 1) | 10-14 09:29-10:06 | samples 1-316, crashed on the first clip > 30 s (exit 1); 300 kept via checkpoints | 2,222 s | yes | $0.80 |
| `fullrun2000_resume` | 10-15 02:11-05:31 | samples 301-2000 | ~12,000 s | **no**: folder deleted with the instance (it held credentials) | ~$4.30 |
| `fullrun2000_finish` | 10-15 05:41 | re-ran the guard; loaded the finished pickle from cache, no inference | 27 s | phases only | $0.01 |
| `fullrun2000_finish_longform` | 10-15 05:42-05:46 | 8 long samples × 6 Whisper models, `WHISPER_LONGFORM=sequential` | 231 s | yes | $0.08 |

So the per-model and per-sample numbers below come from the first 316 samples only (16 % of the run). The resume wall time and rate come from the chat on 2026-10-15, not from files.

## Whole run
- **~4.0 h of container time, ~$5.20** (instance setup and idle time not included).
- Main run: 2,222 s + ~12,000 s for 2016 sample passes ≈ 7.05 s per sample (154 inferences) ≈ **21.8 inf/s, ~8.5 samples/min**. The attempt-1 rate (21.9 inf/s, 7.03 s/sample) predicted 3.9 h for 2000 samples; the real total was 3.95 h, so the rate held for the whole run.
- **Waste: 16 samples × 154 = 2,464 inferences** run twice (attempt 1 got to sample 316, the last checkpoint was at 300), plus 84 samples prepped but never transcribed. Under 1 % of the run.

## Attempt 1 breakdown (316 samples, 48,664 inferences)

| Part | Seconds | Share of wall |
|---|---|---|
| Fixed (start, model loads) | 9.4 | 0.4 % |
| GPU inference | 2,171.5 | 97.7 % |
| Waiting for prep workers | 6.0 (all on sample 1) | 0.3 % |
| Scoring | 12.4 | 0.6 % |
| Checkpoints (6 × 1.4 s) | 8.4 | 0.4 % |
| Other overhead | ~14 | 0.6 % |
| **Wall** | **2,222.2** | |

- **The run is GPU-bound.** GPU share of per-sample work is 0.99, GPU busy in 93 % of the 0.5 s nvidia-smi readings, util p95 100 %, mean power 137 W against the 150 W cap.
- **Prep is fully hidden.** 639 s of prep (p50 1.3 s, max 9.5 s per sample) on 4 workers, with no waiting after the first sample. More workers or thread changes would not speed this run up.
- **RTF 0.0057** (175 s of audio per second), $0.0074 per audio hour, $0.059 per 1k inferences.
- Checkpoints every 50 samples came every 5.4-6.5 min; the rate was steady across the attempt.

## Per model (attempt 1, 316 samples each, 6,952 inferences each)

| Model | Infer s | Share of GPU time | Inf/s | Audio s/s | Sample p50 s | Sample p99 s |
|---|---|---|---|---|---|---|
| whisper-tiny | 189.8 | 8.7 % | 36.6 | 292 | 0.28 | 1.97 |
| whisper-base | 219.4 | 10.1 % | 31.7 | 253 | 0.34 | 2.61 |
| whisper-small | 206.6 | 9.5 % | 33.7 | 268 | 0.57 | 4.05 |
| whisper-medium | 395.2 | 18.2 % | 17.6 | 140 | 1.18 | 2.08 |
| whisper-large-v3 | 660.6 | 30.4 % | 10.5 | 84 | 1.99 | 3.28 |
| whisper-large-v3-turbo | 434.0 | 20.0 % | 16.0 | 128 | 1.35 | 1.62 |
| wav2vec2-large | 66.0 | 3.0 % | 105.4 | 841 | 0.17 | 0.64 |

- **large-v3 is 30 % of the GPU time**; large-v3 + turbo + medium are 69 %. Dropping or shrinking one of those is the only lever that moves total time much.
- **Turbo is 1.5× faster than large-v3** (16.0 vs 10.5 inf/s) for 0.6 points of mean WER (5.9 vs 5.3, from the first look, a file not found in the repo or the checkout; the corpus WER in `results.md` is 5.68 vs 5.16).
- **tiny, base and small cost about the same**, and tiny is barely faster than small. Their medians scale with size, but their tails don't: p99 is 2-4 s, more than medium's. The slowest samples are short (e.g. small takes 4.2 s on a sample with 67 s of audio), which looks like repetition loops running to the token limit. That fits the WER > 100 % hallucination rows in `results.md`. Inferred from timing here; the transcript check in `../v2_readiness_review.md` later found loop rows running to the 448-token limit, which supports it.
- **wav2vec2 (CTC, no decoder) is 10× faster than large-v3** and only 3 % of the run.

## Long-clip side run (sequential long-form)
8 samples × 22 × 6 models = 1,056 inferences in 231 s (186 s inference). 4.9 inf/s and 158 audio s/s; these are 30-35 s clips decoded in 30 s windows with timestamps, so inf/s isn't comparable with the main run. Per model, large-v3 took 6.7 s per sample (p50), turbo 3.7 s. 95 "did not predict an ending timestamp" warnings, harmless as noted in handoff 5. The first sample waited 10 s for prep, since only 8 samples were prepped.

## Against earlier sweeps
Only `bs_sweep1` is still on disk or in GCS. The other sweep folders (`worker_sweep_bs22`, `thread_caps`, `spc_sweep1`, `models7_*`) are gone from both, so these use the numbers in `MEMO.local.md`.

| Sweep | Setting | Result | Full run |
|---|---|---|---|
| `bs_sweep1` (20 samples, 1 model) | batch 1/8/16/22 | BS22 best, 7.4 inf/s, GPU share 0.37 (prep-bound, before the worker pool) | BS22, now GPU-bound |
| `worker_sweep_bs22` | 1/2/4 workers | all equal | 4 workers, prep fully hidden, so this still holds |
| `thread_caps` | none/1/2 | caps 2, -25 % inference | caps 2 |
| `spc_sweep1` | SAMPLES_PER_CALL 1/2/4 | no gain | 1 |
| `models7_100` | 7 models, 100 samples | 23.4 inf/s | **21.9 inf/s, 6 % slower** |

The 6 % gap to `models7_100` is plausibly the different sample mix (longer clips, more hallucination tails in a larger set). It can't be checked without that sweep's timing data.

## Problems found
1. **84 % of the run has no timing data** (the deleted `fullrun2000_resume`). Fixed for future runs by `2591359` on `main` (`7a1b150` on `exploration`) (detached runs, `run.json` guard), but this run's per-sample timing for samples 301-2000 can't be recovered.
2. **VRAM peak 21.8 of 23.0 GB (95 %)** with all 7 models loaded at batch 22. A larger batch, an eighth model, or a longer clip could hit out-of-memory.
3. **`analyze.py` quirks in these reports**:
   - `n_samples` is 2,212 for attempt 1; that's sample records (316 × 7), not samples (known open item from handoff 5).
   - A failed run (exit 1) is left out of `summary.csv` and `report.md`, so the `fullrun2000` report shows only dashes even though `runs.csv` has all its numbers.
   - `fullrun2000_finish` reports "projection for 44,000 inferences", and `fullrun2000` uses 308,000. The target isn't consistent across sweeps of the same run.
   - `analyze.py --combine` (handoff 5, next step 2) can't produce the total because the resume folder is gone. The totals above are by hand.
4. **`bench/TIMING_SPEC.md` is out of date with the code.** `sample` records no longer carry `t_prep_s`; prep is its own `prep` event (one per sample, from the workers), and `sample` adds `t_wait_s`, `group_size` and `ts`. `checkpoint` records carry `checkpoint_idx` and `t_fetch_s` but no `model`. There is also a `finalize` phase.
5. The local pickle is named `...2000samples__fullrun2000.pkl`, but the run label in the logs is `...2000samples__fullrun2000_SAMPLES_PER_CALL_1_r1`. It was renamed after download. That only matters if a script looks it up by label.
6. GCS `transcribe_cache/` is now empty (including `checkpoint/`), so the Mac copy of the two pickles is the only one.

## What would make a full comparison possible
- The full run used a single config, so it can't compare hyperparameters by itself. Comparing configs needs the earlier sweep folders back, or a re-run of a short sweep.
- A per-model timing figure for the whole run needs a rerun. 316 samples is enough for the shares and rates above (checkpoint intervals were steady), but not for tail percentiles over all 2000.
