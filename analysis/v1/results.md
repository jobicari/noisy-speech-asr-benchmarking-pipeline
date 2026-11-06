# v1 (fullrun2000) results

2000 LibriSpeech samples, 22 corruption types (including `none`), 7 models (6 Whisper sizes on HF + wav2vec2-large-960h-lv60-self), NVIDIA A10 on Lambda.

Sources:
- Results: `results/transcribe_cache/fullRun__hf__986ea79c__2000samples__fullrun2000.pkl` (same file as `gs://<bucket>/results/transcribe_cache/fullRun__hf__986ea79c__2000samples__fullrun2000_SAMPLES_PER_CALL_1_r1.pkl`, uploaded 2026-10-15 05:31 UTC).
- Long-clip side run: `fullRun__hf__a9498565__8samples__fullrun2000_finish_longform.pkl`.
- Timing: `results/lambda_bench_runs/fullrun2000`, `fullrun2000_finish`, `fullrun2000_finish_longform`.

## Completeness

- 308,000 rows = 2000 samples x 7 models x 22 corruptions, exactly one row per (sample, model, corruption). No gaps, no duplicates.
- 46,200 rows (the first 300 samples, from the crashed attempt's 6 checkpoints) have no `duration_s`. They predate the field, and since that attempt crashed on the first clip over 30 s, none of them is a long clip. The long-clip detection is therefore complete.
- Empty hypotheses: 134 for wav2vec2 (84 on `WN_3-SV_.5-SDO_8_100_15`, 22 on `BL_500_1000`, 17 on `PL_5_.5`), 1 for turbo, 0 for the others. These are plausible model behaviour on destroyed audio, not pipeline failures.

## Throughput and timing

The run happened in three pieces, and only two have timing:

| Piece | Sweep folder | What it did | Wall | Inf/s |
|---|---|---|---|---|
| First attempt | `fullrun2000` | 300 samples checkpointed (6 x 50), crashed at 10:06 UTC on 2026-10-14 on `2094-142345-0008` (31.6 s, "more than 3000 mel input features") | 2,222 s | 21.9 (steady 22.0) |
| Resume | none | the remaining 1,700 samples; finished and uploaded the final pickle at 05:31 UTC on 2026-10-15 | unknown | unknown |
| Finish | `fullrun2000_finish` | found the final pickle in the cache and exited | 27 s | - |
| Long-clip side run | `fullrun2000_finish_longform` | 8 clips over 30 s x 6 Whisper models x 22 corruptions, `WHISPER_LONGFORM=sequential` | 231 s | 4.87 |

The resume ran in a plain SSH shell that dropped (see the README), so no sweep folder, `timing.jsonl` or GPU/CPU logs were saved for it. **The timing of 85% of the run is missing**, and `analyze.py --combine` cannot produce a total. From the first attempt's rate, the full 308,000 inferences take about 308,000 / 21.9 = 3.9 h of GPU time.

First attempt, per-sample breakdown: GPU inference is 99% of per-sample work (infer 2,171 s, scoring 12 s, waiting on prep 6 s after the first sample). GPU utilisation 76% mean, busy 93% of the time, VRAM peak 21.8 GB of 23 GB, 137 W. The 7 models at batch size 22 nearly fill the A10's memory, so there is little headroom for a larger batch.

## Accuracy

Corpus WER % (sum of S+D+I over sum of reference words):

| Model | Clean (`none`) | All 22 corruptions |
|---|---|---|
| whisper-tiny | 6.91 | 42.16 |
| whisper-base | 4.61 | 32.62 |
| whisper-small | 3.08 | 12.25 |
| whisper-medium | 2.56 | 7.90 |
| whisper-large-v3 | 1.76 | **5.16** |
| whisper-large-v3-turbo | 1.91 | 5.68 |
| wav2vec2-large-lv60-self | **1.59** | 20.00 |

Clean numbers are in line with published LibriSpeech test-clean results, which is a good sanity check of normalisation and scoring.

Per corruption (corpus WER %), sorted by large-v3:

| Corruption | tiny | base | small | medium | large-v3 | turbo | w2v2 |
|---|---|---|---|---|---|---|---|
| none | 6.9 | 4.6 | 3.1 | 2.6 | 1.8 | 1.9 | 1.6 |
| codec_8k | 9.4 | 5.9 | 3.7 | 2.7 | 1.9 | 2.0 | 1.9 |
| PL_40_.1 | 9.6 | 6.1 | 3.5 | 2.5 | 1.9 | 2.0 | 1.8 |
| clip_1 | 14.7 | 9.0 | 4.5 | 3.3 | 2.0 | 2.4 | 2.7 |
| clipN_1 | 14.9 | 8.9 | 4.4 | 3.2 | 2.0 | 2.3 | 2.7 |
| SV_0.5 | 12.0 | 7.0 | 4.2 | 2.9 | 2.1 | 2.2 | 2.2 |
| codec_6k | 11.3 | 7.9 | 4.5 | 3.1 | 2.2 | 2.4 | 2.3 |
| PL_30_.2 | 15.9 | 9.4 | 5.0 | 3.5 | 2.2 | 2.5 | 2.6 |
| PL_20_.3 | 20.4 | 12.3 | 5.8 | 3.9 | 2.5 | 2.6 | 3.1 |
| BL_800_3000 | 32.1 | 15.3 | 5.7 | 3.9 | 2.5 | 2.6 | 3.7 |
| PL_10_.4 | 22.8 | 13.1 | 5.7 | 3.7 | 2.6 | 2.8 | 10.4 |
| clipN_05 | 21.5 | 13.3 | 6.2 | 4.3 | 2.6 | 2.9 | 8.5 |
| clip_05 | 22.5 | 14.3 | 6.2 | 4.3 | 2.6 | 2.9 | 8.5 |
| PL_5_.5 | 26.4 | 16.1 | 6.9 | 4.3 | 2.7 | 3.2 | 83.1 |
| SDO_8_100_15 | 17.2 | 12.6 | 7.3 | 5.4 | 3.2 | 3.6 | 3.4 |
| BL_600_2000 | 74.1 | 28.1 | 9.9 | 5.5 | 3.2 | 3.4 | 9.5 |
| RV_0.5_251 | 26.2 | 15.5 | 7.3 | 4.6 | 3.3 | 3.4 | 8.6 |
| clipN_025 | 34.2 | 22.1 | 9.9 | 5.8 | 3.4 | 3.7 | 34.4 |
| clip_025 | 33.9 | 22.7 | 9.2 | 5.9 | 3.5 | 3.8 | 34.4 |
| WN_3snr | 37.2 | 24.3 | 12.7 | 8.1 | 5.5 | 6.5 | 39.7 |
| WN_3-SV_.5-SDO_8_100_15 | 127.1 | 103.4 | 47.0 | 33.7 | 26.3 | 30.7 | 86.0 |
| BL_500_1000 | 337.4 | 345.8 | 96.8 | 56.7 | 33.4 | 35.0 | 88.9 |

Observations:
- large-v3 is the most robust model on every corruption; turbo is within about 0.5 points except on the two hardest ones. wav2vec2 wins on clean audio only.
- wav2vec2 collapses on `PL_5_.5` (83% vs 2.7% for large-v3), on heavy clipping (`clip_025`, 34%) and on white noise (`WN_3snr`, 40%). Whisper's language model hides short dropouts that a CTC model cannot.
- The tiny and base WER over 300% on `BL_500_1000` is almost all insertions (241% and 260% insertion rate): repetition loops such as "i am so so so so ..." on a 4-word reference. The same happens, less often, on the combined corruption. This inflates corpus WER for small models; a capped or median WER, or a hallucination flag (for example insertions > 2x reference words), would describe them better.
- `BL_500_1000` and `WN_3-SV_.5-SDO_8_100_15` are the only corruptions that hurt the large models much. Most others leave large-v3 under 3.5%.

## Clips over 30 s (8 samples)

| Model | Chunked (main run) | Sequential (side run) |
|---|---|---|
| tiny | 28.6 | 39.0 |
| base | 24.1 | 22.4 |
| small | 18.0 | 12.1 |
| medium | 9.7 | 8.0 |
| large-v3 | 6.8 | 5.9 |
| turbo | 6.3 | 5.7 |

Sequential long-form decoding is better for base and above (by 6 points on small), and worse for tiny, which hallucinates more with timestamps. Only 8 clips x 22 corruptions, so treat these as indicative.

## What looks wrong or incomplete

1. **No timing for the resume**, which did 85% of the work. Throughput for the full run is an extrapolation from the first 300 samples. The detached-run fix (2591359 on `main`, 7a1b150 on `exploration`) prevents this next time.
2. **`fullrun2000_finish` produced no metrics**: it only loaded the cached result, so its report is mostly dashes. That is expected, not a bug, but its report should not be read as a run.
3. **Repetition hallucinations** in tiny/base dominate their corpus WER on two corruptions (see above).
4. 46,200 rows without `duration_s` (harmless, explained above).
