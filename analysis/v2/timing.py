"""v2 (fullrun2000b) timing: phases, per model, over time, GPU, cost, comparison with v1 attempt 1.

Run from the repo root (stdlib + pandas):
    python analysis/v2/timing.py
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

R = Path("results/lambda_bench_runs")
PRICE = 1.29
ORDER = ["tiny", "base", "small", "medium", "large-v3", "turbo", "w2v2"]


def md(df):
    df = df.reset_index()
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def short(m):
    return (m.replace("openai_whisper-", "").replace("facebook_wav2vec2-large-960h-lv60-self", "w2v2")
            .replace("large-v3-turbo", "turbo"))


def run(sweep):
    d = next(p for p in (R / sweep).iterdir() if (p / "run.json").exists())
    meta = json.loads((d / "run.json").read_text())
    recs = [json.loads(line) for line in (d / "timing.jsonl").read_text().splitlines()]
    for r in recs:
        r["t"] = datetime.fromisoformat(r["ts"]).timestamp()
        if "model" in r:
            r["model"] = short(r["model"])
    gpu = []
    with (d / "gpu.csv").open() as f:
        for row in csv.reader(f):
            try:
                gpu.append({"t": datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(),
                            "util": float(row[2]), "mem": float(row[4]), "power": float(row[6])})
            except (ValueError, IndexError):
                pass
    return meta, pd.DataFrame(recs), pd.DataFrame(gpu)


def report(sweep):
    meta, t, gpu = run(sweep)
    wall = meta["t_end"] - meta["t_start"]
    s, p, ph, ck = (t[t.event == e] for e in ("sample", "prep", "phase", "checkpoint"))
    print(f"\n# {sweep}: wall {wall:,.0f} s ({wall / 3600:.2f} h), ${PRICE * wall / 3600:.2f}, exit {meta['exit_code']}")
    print("records:", t.event.value_counts().to_dict())
    rc = t[t.event == "run_config"]
    if len(rc):
        print("versions:", rc.iloc[0]["versions"], "gpu:", rc.iloc[0]["gpu"])

    # Wall-time breakdown
    waits = s.groupby("sample_idx").t_wait_s.max()
    parts = {
        "startup": ph[ph.name == "startup"].seconds.sum(),
        "model loads": ph[ph.name == "model_load"].seconds.sum(),
        "inference": s.t_infer_s.sum(),
        "scoring": s.t_score_s.sum(),
        "waiting for prep": waits.sum(),
        "  of which first sample": waits.iloc[0] if len(waits) else 0,
        "fetching from the stream": ck.t_fetch_s.sum() if "t_fetch_s" in ck else 0,
        "checkpoint saves": ck.seconds.sum(),
        "finalize": ph[ph.name == "finalize"].seconds.sum(),
    }
    accounted = sum(v for k, v in parts.items() if not k.startswith("  "))
    parts["unaccounted (container start, imports, exit)"] = wall - accounted
    print(md(pd.DataFrame({"seconds": parts, "share_pct": {k: 100 * v / wall for k, v in parts.items()}}).round(1)))

    n_inf = s.n_items.sum()
    audio = s.groupby("sample_idx").audio_s.first().sum()
    steady = wall - ph.seconds.sum()
    print(f"inferences {n_inf:,}, audio {audio / 3600:.1f} h (all corruptions, once per sample), "
          f"{n_inf / wall:.2f} inf/s overall, {n_inf / steady:.2f} steady, RTF {steady / audio:.4f} "
          f"(per model {steady / audio / s.model.nunique():.4f}), ${PRICE * steady / audio:.3f} per audio hour "
          f"through all models, ${PRICE * steady / audio / s.model.nunique():.4f} per audio hour per model, "
          f"${PRICE * steady / 3600 / n_inf * 1000:.4f} per 1k inferences")
    print(f"prep: {p.t_prep_s.sum():,.0f} s total on the workers, p50 {p.t_prep_s.median():.2f} s, max {p.t_prep_s.max():.2f} s per sample")

    # Per model
    g = s.groupby("model")
    pm = pd.DataFrame({
        "infer_s": g.t_infer_s.sum(), "share_pct": 100 * g.t_infer_s.sum() / s.t_infer_s.sum(),
        "inf_per_s": g.n_items.sum() / g.t_infer_s.sum(), "audio_s_per_s": g.audio_s.sum() / g.t_infer_s.sum(),
        "p50_s": g.t_infer_s.median(), "p90_s": g.t_infer_s.quantile(.9), "p99_s": g.t_infer_s.quantile(.99),
        "max_s": g.t_infer_s.max(),
    }).reindex([m for m in ORDER if m in g.groups]).round(2)
    print(md(pm))

    # Over time: seconds per checkpoint
    if len(ck) > 1:
        ct = ck.t.diff().dropna()
        print(f"checkpoints: {len(ck)}, interval p50 {ct.median():.0f} s, min {ct.min():.0f}, max {ct.max():.0f}; "
              f"first half {ct[:len(ct) // 2].mean():.0f} s, second half {ct[len(ct) // 2:].mean():.0f} s")
        per_ck = s.assign(ck=s.groupby("model").cumcount() // 50).groupby("ck").t_infer_s.sum()
        print("inference seconds per 50-sample block, by block:", per_ck.round(0).astype(int).tolist())

    # GPU
    if len(gpu):
        print(f"GPU: util mean {gpu.util.mean():.0f}%, p50 {gpu.util.median():.0f}, busy (>10%) {100 * (gpu.util > 10).mean():.0f}% "
              f"of samples, VRAM peak {gpu.mem.max():,.0f} MiB, mean power {gpu.power.mean():.0f} W, "
              f"max {gpu.power.max():.0f} W")
    return s


v2s = report("fullrun2000b")
v1s = report("fullrun2000")
report("fullrun2000b_longform")
report("fullrun2000b_loops")

print("\n# v1 attempt 1 vs v2 on the same samples")
same = sorted(set(v1s.sample_idx) & set(v2s.sample_idx))
a = v1s[v1s.sample_idx.isin(same)].groupby("model").t_infer_s.sum()
b = v2s[v2s.sample_idx.isin(same)].groupby("model").t_infer_s.sum()
cmp = pd.DataFrame({"v1_s": a, "v2_s": b, "v2_over_v1": b / a}).reindex(ORDER).round(3)
print(f"{len(same)} samples in both")
print(md(cmp))
print("total:", round(a.sum()), round(b.sum()), round(b.sum() / a.sum(), 3))
# Same samples, per-sample ratio distribution for the largest model
j = v1s.merge(v2s, on=["model", "sample_idx"], suffixes=("_1", "_2"))
j["r"] = j.t_infer_s_2 / j.t_infer_s_1
print("per sample/model ratio v2/v1: p10 %.2f p50 %.2f p90 %.2f" % tuple(j.r.quantile([.1, .5, .9])))

print("\n# Rest of the run (samples 317-2000) vs first 316 in v2")
first = v2s[v2s.sample_idx.isin(same)]
rest = v2s[~v2s.sample_idx.isin(same)]
print("v2 inf/s on first 316:", round(first.n_items.sum() / first.t_infer_s.sum(), 2),
      "rest:", round(rest.n_items.sum() / rest.t_infer_s.sum(), 2))
print("audio s per sample, first 316:", round(first.groupby("sample_idx").audio_s.first().mean(), 1),
      "rest:", round(rest.groupby("sample_idx").audio_s.first().mean(), 1))

print("\n# Cost of repetition loops in the main run")
import pickle
import zlib
rows = pickle.load(open(R / "fullrun2000b/SAMPLES_PER_CALL_1_r1/fullRun__hf__986ea79c__2000samples__fullrun2000b_SAMPLES_PER_CALL_1_r1.pkl", "rb"))
loops = {}
for r in rows:
    h = (r["hyp"] or "").encode()
    if h and len(h) / len(zlib.compress(h)) > 2.4 and "whisper" in r["model"]:
        key = (r["idx"], short(r["model"]))
        loops[key] = loops.get(key, 0) + 1
v2s["loops"] = [loops.get((i, m), 0) for i, m in zip(v2s.sample_idx, v2s.model)]
g = v2s[v2s.model != "w2v2"].assign(has=lambda d: d.loops > 0).groupby(["model", "has"]).t_infer_s
lc = pd.DataFrame({"calls": g.size(), "mean_s": g.mean(), "total_s": g.sum()}).round(2)
print(md(lc))
base = v2s[(v2s.loops == 0)].groupby("model").t_infer_s.mean()
extra = sum((row.t_infer_s - base[row.model]) for row in v2s[v2s.loops > 0].itertuples())
print(f"extra inference time on sample calls with a loop (vs that model's loop-free mean): {extra:,.0f} s "
      f"({100 * extra / v2s.t_infer_s.sum():.1f}% of all inference)")
