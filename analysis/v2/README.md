# v2: the second full run (`fullrun2000b`)

v2 is the rerun of [v1](../v1/README.md), planned in [`v2_plan.md`](v2_plan.md). It ran on 2026-10-29 on one Lambda A10 from `main` at `5fe7566`, in one piece, with complete timing.

Details live next to this file:
- [`results.md`](results.md): data check, v1 agreement, WER by model and corruption, loops, the fallback pass, long clips, clip length.
- [`timing.md`](timing.md): wall time, where the time went, per-model GPU time, the GPU time lost to loops, the side passes, the comparison with v1.
- [`results.py`](results.py) and [`timing.py`](timing.py) compute every number in them. Run them from the repo root in the `asr-benchmark-pipe` env after downloading the three sweep folders into `results/lambda_bench_runs/`.

## What was run

| Piece | What | Wall |
|---|---|---|
| Main run | 2000 LibriSpeech test-clean samples x 22 corruptions x 7 models = 308,000 transcriptions, greedy decoding | 4.38 h |
| Long-clip pass | the 8 samples with a clip over 30 s, 6 Whisper models, sequential long-form decoding | 4 min |
| Looped-clip pass | the 1,077 clips where a Whisper model looped, 6 Whisper models, temperature fallback | 47 min |
| **Total** with finders, uploads and terminate | | **about 5.3 h** |

Same settings as v1 (`SEED=123`, `BATCH_SIZE=22`, 4 prep workers, thread caps 2, fp16, chunked long-form), with library versions pinned to v1's (transformers 5.19.0, torch 2.14.1+cu126). Results and logs are in GCS under `lambda_bench_runs/fullrun2000b{,_longform,_loops}/`, each run's results pickle next to its logs, and downloaded locally to `results/lambda_bench_runs/` for this analysis.

## Main findings

1. **v2 reproduces v1 exactly.** All 308,000 transcripts are identical to v1's, for every model. The pipeline is deterministic, and every v1 accuracy number holds.

2. **large-v3 is the most robust model**: 5.16% WER over all corruptions, best on 19 of 22 corruptions, and only 2.9x its clean WER. Turbo is close (5.68%) at 1.5x the speed.

   | Model | Clean | All corruptions | Capped |
   |---|---|---|---|
   | tiny | 6.91 | 42.16 | 27.33 |
   | base | 4.61 | 32.62 | 18.86 |
   | small | 3.08 | 12.25 | 10.98 |
   | medium | 2.56 | 7.90 | 7.69 |
   | large-v3 | 1.76 | **5.16** | **5.08** |
   | turbo | 1.91 | 5.68 | 5.43 |
   | wav2vec2 | **1.59** | 20.00 | 19.99 |

3. **wav2vec2 wins on clean audio and collapses under packet loss, clipping and noise** (83%, 34% and 40% WER, against 3-6% for large-v3). Half its errors are deletions, while Whisper's are insertions and substitutions.

4. **Repetition loops drive the small models' WER**: 671 tiny and 504 base transcripts loop, 92% of them on the 500-1000 Hz band and the combined noise chain. Capping each row at 100% gives the realistic numbers (tiny 27%, base 19%).

5. **Temperature fallback breaks 96-100% of loops but doesn't recover the speech.** Looped rows drop from about 900% to about 100% WER, but capped WER only improves from about 100% to 92-94%. With fallback, the whole-run WER lands within 0.5 points of capped greedy WER for every model. **Report capped WER next to corpus WER**: it is what Whisper with its usual safety net would score.

6. **Sequential long-form decoding beats chunking for base and up** on clips over 30 s, and is worse for tiny. Same as v1, and only 8 samples.

7. **Short clips are hardest for Whisper** (large-v3 8.3% under 5 s, 3.5% at 15-30 s). wav2vec2 doesn't care about length.

8. **Timing is complete, and the run is GPU-bound**: inference is 98.3% of wall time, prep is fully hidden, and the rate held steady for 4.4 h. large-v3 alone is 28% of GPU time.

9. **Loops take 15% of all GPU time.** They take 73% of tiny's time and 66% of base's, which is why tiny is slower than small.

10. **v2 was 10% slower than v1 on the same samples.** The slowdown is mostly in the small models (+19-23%, large-v3 +4%), with lower GPU utilisation. That points at more CPU-side overhead per decoding step, from either v1's real library versions (never recorded) or the instance. Not proven; a 10-minute A/B on one instance would settle it.

## What to watch for a v3

- **VRAM peaked at 21.8 GiB (22,333 MiB) of 22.06 GiB usable.** An eighth model, a bigger batch or a newer transformers (5.21.0 ran out of memory in the smoke test) needs a smoke test on a looping sample first.
- **The fallback pass is slow** (2.3 inf/s, 47 min for 2% of the transcriptions), because each looping clip is decoded to the token limit once and then up to 5 more times. Running fallback in the main run instead would not save the loop time: the greedy attempt still runs first. Keep it as a side pass, or skip it and report capped WER, which gives nearly the same numbers.
- **Versions are now logged** in every run's `run_config` record, and `sweep.json` records the code and image commits. A v3 can be compared with v2 exactly.

## Figures

[`figures.py`](figures.py) draws the README's figures into `docs/figures/` from the main run's results pickle:

```bash
python analysis/v2/figures.py
```

- `robustness_vs_size.png`: clean and corrupted WER against Whisper model size, with wav2vec2 for contrast
- `wer_by_corruption.png`: capped WER for every model and corruption
- `accuracy_vs_speed.png`: WER against transcriptions per second on the A10
- `loop_gpu_time.png`: GPU time per Whisper model, and how much of it repetition loops add
- `pipeline_speedups.png`: wall time of the same small run through each tuning step
