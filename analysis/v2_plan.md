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

Already on `main` (2591359): the run wrapper detaches with `setsid nohup` and containers run with `docker run -d` and `docker wait`, so a dropped SSH session no longer loses a run's logs. That was the cause of v1's timing gap.

**Per-sample timing should now cover all 2000 samples.** The code writes a `sample` record per sample per model and a `prep` record per sample, and with the detach fix nothing should be lost. Expected in `timing.jsonl`: 2,000 `prep`, 14,000 `sample`, 40 `checkpoint`, 7 `model_load`, one `startup`, one `run_config`, one `finalize`. This is from reading the code; it is not verified by a run.

## Decoding: keep the main decode unchanged

Decided: v2's main pass decodes exactly as v1 did (greedy, no temperature fallback, no compression-ratio check), so every v2 number is comparable with v1. The analysis reports **capped WER** (each row's errors capped at its reference length) and a loop flag next to corpus WER, as in the v1 table. The alternative, turning on OpenAI-style temperature fallback for the whole run, was not chosen: it would change every Whisper number and slow the looping models.

## Second pass: rerun the looping clips with temperature fallback

Planned: after the main pass, rerun only the (sample, model, corruption) rows whose decode hit the 448-token limit, this time with temperature fallback. This shows how much of the loop problem a standard Whisper setup removes, without touching the comparable main results.

Not designed or implemented yet. Open points:

- How to select the rows. v1 loop rows had 440-530 hypothesis words; a token count at or near the limit is the direct test, but the pipeline does not record token counts today.
- Which fallback settings (temperature schedule, compression-ratio and log-prob thresholds). OpenAI's `transcribe()` defaults are the obvious start; not decided.
- Where the results go: a separate result file and run label, so the main pickle stays a clean v1 comparison.
- Expected size: in v1, about 860 rows across all models (435 tiny, 374 base, 39 small, 7 medium, 1 large-v3, 6 turbo). The v2 count is unknown until the main pass is done.

## Open decisions for Jo

| # | Decision | Recommendation |
|---|---|---|
| B | Stall alert: a background check in `run_and_terminate.sh` that sends an ntfy alert if no container log line appears for 30 minutes. Alert only, no kill | Yes |
| C | After the main sweep, copy the final results pickle into the sweep's GCS folder, so results and timing live together and survive a cleanup of `transcribe_cache/` | Yes |
| D | Upload the Mac's `bs_sweep1` folder to GCS (the only earlier sweep left) | Yes; rerunning the other short sweeps would need a GPU instance |
| - | Run label `fullrun2000b` for `SWEEP_ID` and `RUN_POSTFIX`. Already in the README on the readiness branch; reusing `fullrun2000` would overwrite v1's GCS logs and could load stale results | Confirm |

None of B, C or D is implemented on any branch.

## Known gaps

- **`evaluation.run_framework_speed_comparison` is broken**: it calls `load_transcriber_engine` with an old signature. It is not on the run path (nothing in `app.py`, `src/` or the scripts imports `evaluation.py`, and the `RUN_FRAMEWORK_SPEED_COMPARISON_TEST` flag is read by nothing). Left for a separate fix.
- **VRAM is at 21.8 of 23 GB.** The same config passed all 2000 samples in v1, so it should pass again, but any change (an eighth model, a bigger batch) needs a smoke test first.
- **A hang runs up the bill with no alert** until decision B is done.
- **Earlier sweeps cannot be compared** with v2 except through the numbers kept in the README tables; their folders are gone.
- **Hub streaming errors** mid-run crash the run. Accepted: resume handles it.

## Before and after the run

The checklist for trusting v2 (clean checkout, new label, exit status, 308,000 rows, timing record counts, `run_config` values, clean WER within a few tenths of v1, long-clip side run) is in the last section of [`v2_readiness_review.md`](v2_readiness_review.md).
