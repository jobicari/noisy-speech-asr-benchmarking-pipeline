#!/usr/bin/env python3
"""Write the demo page's audio: the clean and corrupted version of each example clip.

Run from the repo root in the project env, with the v2 results pickle downloaded
(see analysis/v2/README.md), and network access to the Hugging Face Hub:

    SEED=123 PYTHONPATH=src python scripts/demo/export_demo_clips.py

It regenerates each corruption with the same seeds as fullrun2000b, so the audio is the
audio the models heard. As a check, it recomputes the audio-health metrics of every
corrupted clip and compares them with the ones stored in the results pickle.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if os.getenv("SEED") != "123":
    sys.exit("Set SEED=123, the seed of fullrun2000b")

from asr_pipeline_benchmark.audio_analysis import analyze_asr_audio_health  # noqa: E402
from asr_pipeline_benchmark.data import get_speech_dataset  # noqa: E402
from asr_pipeline_benchmark.preprocessing import (  # noqa: E402
    CorruptionListFull, extract_ir_from_mit_dataset)
from asr_pipeline_benchmark.utility import _seed  # noqa: E402

CLIPS = json.loads(Path("docs/demo/clips.json").read_text())  # [{"id", "corr_type"}, ...]
OUT = Path("docs/demo/audio")
V2 = Path("results/lambda_bench_runs/fullrun2000b/SAMPLES_PER_CALL_1_r1/"
          "fullRun__hf__986ea79c__2000samples__fullrun2000b_SAMPLES_PER_CALL_1_r1.pkl")


def numbers(d, prefix=""):
    """Flatten the numeric leaves of a nested dict"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(numbers(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out[prefix + k] = float(v)
    return out


def main():
    rows = pickle.load(open(V2, "rb"))
    stored = {(r["idx"], r["corr_type"]): r for r in rows
              if r["model"] == "openai_whisper-large-v3"}
    wanted = {c["id"] for c in CLIPS}
    ir, _ = extract_ir_from_mit_dataset(ir_idx=251)
    corruptions = CorruptionListFull(ir).corruptions
    OUT.mkdir(parents=True, exist_ok=True)

    found = 0
    for sample in get_speech_dataset().filter(lambda s: s["id"] in wanted):
        sid = sample["id"]
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        sf.write(OUT / f"{sid}__none.wav", audio, sr, subtype="PCM_16")
        for clip in (c for c in CLIPS if c["id"] == sid):
            corr = clip["corr_type"]
            np.random.seed(_seed(sid, corr))
            y = corruptions[corr](audio)
            # Keep the level the model heard; only scale down a clip that would clip in 16-bit WAV
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            sf.write(OUT / f"{sid}__{corr}.wav", y / peak if peak > 1 else y, sr, subtype="PCM_16")
            ref = stored[(sid, corr)]
            dur_ok = abs(len(y) / sr - ref["duration_s"]) < 1e-3
            a, b = numbers(analyze_asr_audio_health(y, sr)), numbers(ref["sample_analysis"])
            diffs = [k for k in a.keys() & b.keys() if not np.isclose(a[k], b[k], rtol=1e-4, atol=1e-6)]
            status = "OK" if dur_ok and not diffs else f"MISMATCH duration_ok={dur_ok} {diffs[:5]}"
            print(f"{sid} {corr}: {status}")
            found += 1
        if found == len(CLIPS):
            break
    if found != len(CLIPS):
        sys.exit(f"Only {found} of {len(CLIPS)} clips found")


if __name__ == "__main__":
    main()
    os._exit(0)  # don't close the streaming iterator (see README, Known gotchas)
