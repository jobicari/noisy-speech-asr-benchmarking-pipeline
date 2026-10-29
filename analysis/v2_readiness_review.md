# v2 (fullrun2000b) readiness review

Branch `run/full-rerun-readiness` (from `main` at 92ad1b4), pushed, no PR yet. Reviewed: `app.py`, `src/`, `run_sweep.sh`, `run_and_terminate.sh`, `progress.sh`, `analyze.py`, `find_long_ids.py`, `upload_results.py`, `Dockerfile`, README. Nothing was run on a GPU or in a container. `exploration` has the same code as `main` (it differs only in the notebook).

## Commits

| Commit | What | Why |
|---|---|---|
| 62d933c | `phase` records for `startup` (dataset, IR, pool) and `resume_skip` (fast-forward on resume), and a `run_config` record (every Config field except the bucket, thread caps, `SAMPLE_IDS`, library versions, GPU name) | Startup and the resume fast-forward were counted as unexplained overhead. The effective settings of a run were not logged anywhere, only inferable from the command line |
| 33edd4a | `analyze.py` counts distinct samples in `n_samples` | It counted one record per model (2,212 instead of 316 for attempt 1) |
| 8549708 | `sweep.json` records the git commit, a dirty flag and the image id | The image has no `.git`, so a sweep could not be traced back to its code |
| 0e87f67 | README: clone `main` (not `exploration`), use `fullrun2000b` as `SWEEP_ID` and `RUN_POSTFIX`, update the estimate to ~4 h / $5.20, document the new records | Reusing `fullrun2000` would overwrite the first run's GCS logs, and the same label could resume from or load stale results |

`run_config` only reads the GPU name if torch is already loaded, because importing torch next to MLX aborts on macOS (two OpenMP runtimes). I checked that on the Mac.

## Failure risks and gaps

| # | Risk or gap | Status |
|---|---|---|
| 1 | Resume lost its SSH session and its timing logs | Already fixed on `main` (2591359): the wrapper detaches with `setsid nohup` and containers run with `docker run -d` and `docker wait`. A crash leaves the instance up (`TERMINATE_ON_FAILURE=0`), and a resume under a new `SWEEP_ID` gets its own folder, which `analyze.py --combine` adds up |
| 2 | Per-sample timing only for 316 of 2000 samples | Was a result of #1. The code logs a `sample` record per sample per model and a `prep` record per sample, so a clean run has all of it |
| 3 | Clips over 30 s crashed attempt 1 | Fixed (`WHISPER_LONGFORM=chunked`). The resume went through all 2000 samples with it |
| 4 | Startup, resume fast-forward and run settings not timed or logged | Fixed (62d933c) |
| 5 | `n_samples` wrong in reports | Fixed (33edd4a) |
| 6 | No link from a sweep to its code | Fixed (8549708) |
| 7 | Run label reuse: a cached result or stale checkpoint with the same label is loaded or resumed silently, and the guard still reports success | Avoided by the new label in the README. `progress.sh` flags a cache load |
| 8 | VRAM peak 21.8 of 23 GB | Not changed. The same data and seed passed all 2000 samples, so it should pass again, but any change (an eighth model, a bigger batch) needs a smoke test first |
| 9 | A hang (e.g. a killed prep worker blocks `Pool.imap` forever, or a stuck GCS call) runs up the bill with no alert | Open, needs a decision (B below) |
| 10 | The final results pickle lives only in `transcribe_cache/` on GCS; the sweep upload holds logs only. That folder is empty now, and the Mac has the only copy | Open, needs a decision (C below) |
| 11 | Earlier sweep folders (`worker_sweep_bs22`, `thread_caps`, `spc_sweep1`, `models7_*`) are gone; only `bs_sweep1` remains | Can't be fixed in code. Their numbers survive in the README tables only |
| 12 | Hub streaming error mid-run crashes the run | Accepted: the resume path handles it |
| 13 | `evaluation.run_framework_speed_comparison` calls `load_transcriber_engine` with an old signature and is broken | Not on the run path: nothing in `app.py`, `src/` or the scripts imports `evaluation.py`, and the `RUN_FRAMEWORK_SPEED_COMPARISON_TEST` flag is read by nothing. Left for a separate fix |

## Repetition loops (tiny and base over 300% WER on BL_500_1000)

Checked on the full 308,000-row pickle. Scoring is correct: `jiwer` counts are right, and references and hypotheses go through the same normalizer. The inflation is real model output. Rows with insertions over 2x the reference length have 440 to 530 hypothesis words, so the decoder runs to its 448-token limit ("i am going to say that i am going to say that ...").

| Model | Loop rows | Share of all errors | Corpus WER | WER with per-row errors capped at reference length |
|---|---|---|---|---|
| tiny | 435 | 37% | 42.2% | 27.3% |
| base | 374 | 45% | 32.6% | 18.9% |
| small | 39 | 10% | 12.3% | 11.0% |
| medium | 7 | 2% | 7.9% | 7.7% |
| large-v3 | 1 | 1% | 5.2% | 5.1% |
| turbo | 6 | 4% | 5.7% | 5.4% |
| wav2vec2 | 0 | 0% | 20.0% | 20.0% |

On `BL_500_1000` alone, capping takes tiny from 337% to 98% and base from 346% to 96%. 90% of loop rows are on `BL_500_1000` and the combined noise+reverb+dropout corruption. These loops also explain the long p99 times of tiny, base and small.

The pipeline decodes greedily with no temperature fallback or compression-ratio check, which is the Hugging Face default but not what OpenAI's reference `transcribe()` does. That is why small models loop freely.

## Decisions for Jo

Status on 2026-10-29: A is decided (option 1, plus a second pass on the looping clips with temperature fallback). B, C and D are still open. See `v2_plan.md`.

- **A. Repetition loops.** (1, recommended) Keep decoding as is, so the rerun is comparable with fullrun2000, and report a loop flag and capped WER next to corpus WER in the analysis. (2) Turn on OpenAI-style temperature fallback with a compression-ratio threshold before the rerun. It describes real Whisper use better, but it changes every Whisper number and slows the looping models.
- **B. Stall alert (recommended).** A background check in `run_and_terminate.sh` that sends an ntfy alert if no container log line has appeared for 30 minutes. Alert only, no kill.
- **C. Keep the results with the logs (recommended).** After the main sweep, copy the final results pickle into the sweep's GCS folder, so results and timing live together and survive a cleanup of `transcribe_cache/`.
- **D. Earlier sweeps.** Upload the Mac's `bs_sweep1` folder to GCS (only `bs_sweep1` still exists). Rerunning the short sweeps would need a GPU instance.

## What has to be true for the rerun to be trusted

1. The instance runs the commit named in `sweep.json` with `git_dirty: false`, and the image was built from that checkout (the `image_id` matches `docker images`).
2. `docker ps` is empty before the start; `gpu.csv` shows no other process holding VRAM.
3. `RUN_POSTFIX` is new (`fullrun2000b_...`) and `container.log` has no "Loaded test results from cache" and no "Resuming" line, unless a resume was intended.
4. Main run: exit 0, `RUN COMPLETED SUCCESSFULLY`, and `pipeline.log` says `Main sweep succeeded`, with both folders `VERIFIED` in GCS.
5. Results: 308,000 rows, exactly one per (sample, model, corruption), `duration_s` on every row.
6. Timing: `timing.jsonl` has 2,000 `prep` records, 14,000 `sample` records (2000 x 7), 40 `checkpoint` records, 7 `model_load`, one `startup`, one `run_config`, one `finalize`. Summed `t_infer_s` per model should be close to the shares measured on attempt 1.
7. `run_config` shows `BATCH_SIZE=22`, `N_PREP_WORKERS=4`, `SAMPLES_PER_CALL=1`, `WHISPER_LONGFORM=chunked`, thread caps 2, the 7 models and the A10.
8. WER on `none` within a few tenths of fullrun2000's (same seed and data). A larger gap means a library or code change, which `run_config.versions` will show.
9. Long-clip side run: 8 samples, 6 Whisper models, `side=1` in the final log line.
