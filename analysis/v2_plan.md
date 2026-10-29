# v2 plan: the second full run (`fullrun2000b`)

v2 is the planned rerun of v1 ([`v1/README.md`](v1/README.md)) with the same data, seed, models and settings. Its main purpose is complete timing: v1 lost per-sample timing for samples 301-2000. Written 2026-10-29. Nothing below has been run on a GPU yet.

The full review behind this plan, with every risk checked and the trust checklist, is in [`v2_readiness_review.md`](v2_readiness_review.md).

## Readiness fixes

On branch `run/full-rerun-readiness` (from `main` at 92ad1b4), pushed, no pull request yet:

| Commit | Change |
|---|---|
| 62d933c | Log `startup` and `resume_skip` phases and a `run_config` record (effective config, library versions, GPU name) |
| 33edd4a | `analyze.py` counts distinct samples in `n_samples` (it counted one per model) |
| 8549708 | `sweep.json` records the git commit, a dirty flag and the image id |
| 0e87f67 | README: clone `main`, use `fullrun2000b` as `SWEEP_ID` and `RUN_POSTFIX`, estimate ~4 h / $5.20 |
| b61fabc | ntfy alert when no run log has grown for `STALL_MIN` minutes (default 30); alert only (decision B) |
| c11ce29 | Each run's results pickle is copied inside GCS next to its logs and its size verified (decision C) |
| 4deb5fa | `CLIPS` (run only listed sample/corruption pairs) and `WHISPER_FALLBACK` (temperature fallback) |
| e3c43ff | Looped-clip second pass in `run_and_terminate.sh` (`find_loop_clips.py`, sweep `<SWEEP_ID>_loops`), README docs and a smoke test |

Already on `main` (2591359): the run wrapper detaches with `setsid nohup` and containers run with `docker run -d` and `docker wait`, so a dropped SSH session no longer loses a run's logs. That was the cause of v1's timing gap.

**Per-sample timing should now cover all 2000 samples.** The code writes a `sample` record per sample per model and a `prep` record per sample, and with the detach fix nothing should be lost. Expected in `timing.jsonl`: 2,000 `prep`, 14,000 `sample`, 40 `checkpoint`, 7 `model_load`, one `startup`, one `run_config`, one `finalize`. This is from reading the code; it is not verified by a run.

## Decoding: keep the main decode unchanged

Decided: v2's main pass decodes exactly as v1 did (greedy, no temperature fallback, no compression-ratio check), so every v2 number is comparable with v1. The analysis reports **capped WER** (each row's errors capped at its reference length) and a loop flag next to corpus WER, as in the v1 table. The alternative, turning on OpenAI-style temperature fallback for the whole run, was not chosen: it would change every Whisper number and slow the looping models.

## Second pass: rerun the looping clips with temperature fallback

Implemented (4deb5fa, e3c43ff), not yet run on a GPU. After the main pass, `run_and_terminate.sh`:

1. Runs `find_loop_clips.py` on the main results: a Whisper transcript counts as a loop when its text compresses more than 2.4x with zlib, the test OpenAI's Whisper uses to trigger fallback. Token counts are not recorded, so this replaces the "hit the 448-token limit" test. On v1 it finds 1,235 transcripts, 1,077 clips in 959 samples; it catches 859 of the 862 runaway rows plus digit loops ("000000...") that the normalizer squashes into one word and that barely show in WER.
2. Reruns those (sample, corruption) clips on all 6 Whisper models with `WHISPER_FALLBACK=True`: OpenAI `transcribe()` defaults, temperatures 0 to 1 in steps of 0.2, compression ratio 1.35 in token space (2.4 on text), log-prob -1.0, seeded from `SEED`. Same audio as the main pass (corruptions are seeded per sample), same chunked long-form handling, `SAMPLES_PER_CALL=20` to batch the clips.
3. Writes a separate sweep `<SWEEP_ID>_loops` with its own run label and results pickle, so the main pickle stays a clean v1 comparison. Both transcripts of each looped clip can then be compared.

Not tested yet: the fallback decoding itself, because torch aborts on import on Jo's Mac. The README has a loop smoke test (`smoke_loops`) to run on the instance before the full run.

## Decisions

| # | Decision | Status |
|---|---|---|
| A | Main decode unchanged; capped WER in the analysis; second pass on looping clips with fallback | Decided, implemented (see above) |
| B | Stall alert in `run_and_terminate.sh` | Done (b61fabc) |
| C | Copy each run's results pickle next to its logs in GCS | Done (c11ce29) |
| D | Upload `bs_sweep1` from the Mac | Done by Jo, 45 files verified in `lambda_bench_runs/bs_sweep1/` |
| E | Pin library versions: build the image once, run the smoke tests, then pin what it installed in `requirements-docker.txt` and the Dockerfile. Today nothing is pinned, so a rebuild pulls the newest torch, transformers and datasets, and v1's versions were never recorded | Open, recommended |
| - | Run label `fullrun2000b` for `SWEEP_ID` and `RUN_POSTFIX` | In the README on the readiness branch |

## Known gaps

- **`evaluation.run_framework_speed_comparison` is broken**: it calls `load_transcriber_engine` with an old signature. It is not on the run path (nothing in `app.py`, `src/` or the scripts imports `evaluation.py`, and the `RUN_FRAMEWORK_SPEED_COMPARISON_TEST` flag is read by nothing). Left for a separate fix.
- **VRAM is at 21.8 of 23 GB.** The same config passed all 2000 samples in v1, so it should pass again, but any change (an eighth model, a bigger batch) needs a smoke test first.
- **Earlier sweeps cannot be compared** with v2 except through the numbers kept in the README tables; their folders are gone.
- **Hub streaming errors** mid-run crash the run. Accepted: resume handles it.

## Before and after the run

The checklist for trusting v2 (clean checkout, new label, exit status, 308,000 rows, timing record counts, `run_config` values, clean WER within a few tenths of v1, long-clip side run) is in the last section of [`v2_readiness_review.md`](v2_readiness_review.md).
