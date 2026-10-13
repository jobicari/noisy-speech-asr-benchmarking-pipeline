#!/usr/bin/env python3
"""Summarise a sweep produced by run_sweep.sh and rank the configs.

Stdlib only, so it runs on the Lambda host or a laptop without installs.

    python3 analyze.py bench_runs/sweep_XXXX --price-per-hour 0.75 \
        --target-inferences 44000

Writes summary.csv, report.md and report.html into the sweep directory.
"""
import argparse
import csv
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

M1_BASELINE_S_PER_INF = 0.98  # M1 Pro full run, from HANDOFF.md


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values):
    return statistics.fmean(values) if values else None


def read_jsonl(path):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_gpu(path):
    rows = []
    if not path.exists():
        return rows
    for r in csv.reader(path.open()):
        if len(r) < 7:
            continue
        try:
            ts = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append({"t": ts, "util": float(r[2]), "mem_util": float(r[3]),
                         "mem_used": float(r[4]), "mem_total": float(r[5]),
                         "power": float(r[6]) if r[6].strip() not in ("[N/A]", "") else None})
        except ValueError:
            continue
    return rows


def read_cpu(path):
    vals = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split(",")
            try:
                vals.append(float(parts[1].strip().rstrip("%")))
            except (IndexError, ValueError):
                pass
    return vals


def analyse_run(run_dir, price):
    meta = json.loads((run_dir / "run.json").read_text())
    wall = meta["t_end"] - meta["t_start"]
    timing = read_jsonl(run_dir / "timing.jsonl")
    samples = [t for t in timing if t.get("event") == "sample"]
    phases = [t for t in timing if t.get("event") == "phase"]
    ckpts = [t for t in timing if t.get("event") == "checkpoint"]
    gpu = read_gpu(run_dir / "gpu.csv")
    cpu = read_cpu(run_dir / "cpu.csv")

    r = {"run_id": meta["run_id"], "config": meta["config"], "repeat": meta["repeat"],
         "exit_code": meta["exit_code"], "wall_s": wall}

    n_items = sum(s.get("n_items", 0) for s in samples) or meta.get("n_inferences")
    audio_s = sum(s.get("audio_s", 0) for s in samples) or None
    prep = sum(s.get("t_prep_s", 0) for s in samples)
    infer = sum(s.get("t_infer_s", 0) for s in samples)
    score = sum(s.get("t_score_s", 0) for s in samples)
    fixed = sum(p.get("seconds", 0) for p in phases)
    work = prep + infer + score
    r.update({
        "n_samples": len(samples) or None, "n_inferences": n_items, "audio_s": audio_s,
        "fixed_s": fixed if phases else None,
        "prep_s": prep if samples else None, "infer_s": infer if samples else None,
        "score_s": score if samples else None,
        "checkpoint_s": sum(c.get("seconds", 0) for c in ckpts) if ckpts else None,
        "overhead_s": wall - work - fixed if samples else None,
        "gpu_share_of_work": infer / work if work else None,
    })

    # Throughput. "steady" excludes one-off phases (model load, dataset init)
    # so it is what dominates a very long run.
    steady_wall = wall - fixed
    r["inf_per_s"] = n_items / wall if n_items else None
    r["steady_inf_per_s"] = n_items / steady_wall if n_items and steady_wall > 0 else None
    r["s_per_inf"] = 1 / r["steady_inf_per_s"] if r["steady_inf_per_s"] else None
    r["speedup_vs_m1"] = M1_BASELINE_S_PER_INF / r["s_per_inf"] if r["s_per_inf"] else None
    r["audio_s_per_s"] = audio_s / steady_wall if audio_s else None
    r["rtf"] = steady_wall / audio_s if audio_s else None
    r["rtf_infer_only"] = infer / audio_s if audio_s and samples else None

    per_sample = [s.get("t_prep_s", 0) + s.get("t_infer_s", 0) + s.get("t_score_s", 0) for s in samples]
    per_infer = [s["t_infer_s"] for s in samples if "t_infer_s" in s]
    for p in (50, 90, 99):
        r[f"sample_p{p}_s"] = pct(per_sample, p)
        r[f"infer_p{p}_s"] = pct(per_infer, p)

    utils = [g["util"] for g in gpu]
    r["gpu_util_mean"] = mean(utils)
    r["gpu_util_p95"] = pct(utils, 95)
    r["gpu_busy_frac"] = sum(u > 10 for u in utils) / len(utils) if utils else None
    r["gpu_mem_peak_mib"] = max((g["mem_used"] for g in gpu), default=None)
    r["gpu_mem_total_mib"] = gpu[0]["mem_total"] if gpu else None
    powers = [g["power"] for g in gpu if g["power"] is not None]
    r["gpu_power_mean_w"] = mean(powers)
    r["cpu_pct_mean"] = mean(cpu)  # docker stats: 100% = one core

    if price:
        r["run_cost_usd"] = price * wall / 3600
        r["usd_per_audio_hour"] = price * steady_wall / audio_s if audio_s else None
        r["usd_per_1k_inf"] = price * steady_wall / n_items * 1000 if n_items else None
    return r


AGG_FIELDS = ["wall_s", "steady_inf_per_s", "s_per_inf", "speedup_vs_m1", "audio_s_per_s", "rtf",
              "rtf_infer_only", "gpu_share_of_work", "prep_s", "infer_s", "overhead_s",
              "sample_p50_s", "sample_p90_s", "sample_p99_s", "infer_p50_s", "infer_p99_s",
              "gpu_util_mean", "gpu_util_p95", "gpu_busy_frac", "gpu_mem_peak_mib",
              "gpu_power_mean_w", "cpu_pct_mean", "usd_per_audio_hour", "usd_per_1k_inf",
              "fixed_s"]


def aggregate(runs, price, target):
    by_cfg = {}
    for r in runs:
        by_cfg.setdefault(r["config"], []).append(r)
    out = []
    for cfg, rs in by_cfg.items():
        ok = [r for r in rs if r["exit_code"] == 0]
        a = {"config": cfg, "runs": len(rs), "failed": len(rs) - len(ok)}
        for f in AGG_FIELDS:
            vals = [r[f] for r in ok if r.get(f) is not None]
            a[f] = mean(vals)
            if f == "steady_inf_per_s" and len(vals) > 1:
                a["steady_inf_per_s_spread"] = (max(vals) - min(vals)) / mean(vals)
        if target and a["steady_inf_per_s"]:
            hours = (target / a["steady_inf_per_s"] + (a["fixed_s"] or 0)) / 3600
            a["proj_hours"] = hours
            a["proj_usd"] = hours * price if price else None
        out.append(a)
    out.sort(key=lambda a: -(a["steady_inf_per_s"] or 0))
    return out


def fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


COLS = [("config", "Config", 0), ("steady_inf_per_s", "Inf/s", 2), ("s_per_inf", "s/inf", 3),
        ("speedup_vs_m1", "× M1", 1), ("audio_s_per_s", "Audio s/s", 1), ("rtf", "RTF", 4),
        ("gpu_share_of_work", "GPU share", 2), ("sample_p50_s", "Sample p50 s", 2),
        ("sample_p99_s", "Sample p99 s", 2), ("gpu_util_mean", "GPU util %", 0),
        ("gpu_mem_peak_mib", "VRAM peak MiB", 0), ("cpu_pct_mean", "CPU %", 0),
        ("usd_per_audio_hour", "$/audio h", 4), ("proj_hours", "Proj. h", 1),
        ("proj_usd", "Proj. $", 2), ("failed", "Failed", 0)]


def write_md(path, sweep, aggs, runs, args):
    lines = [f"# Sweep {sweep.get('sweep_id', '')}", "",
             f"GPU {sweep.get('gpu', '?')}, {sweep.get('host_cpus', '?')} host CPUs, "
             f"image `{sweep.get('image', '?')}`, price ${args.price_per_hour}/h, "
             f"projection for {args.target_inferences:,} inferences.", ""]
    best = aggs[0] if aggs else None
    if best and best["steady_inf_per_s"]:
        lines += [f"**Fastest: `{best['config']}`** at {fmt(best['steady_inf_per_s'])} inferences/s "
                  f"({fmt(best['speedup_vs_m1'], 1)}× the M1 baseline)."]
        if len(aggs) > 1 and aggs[1]["steady_inf_per_s"]:
            gain = best["steady_inf_per_s"] / aggs[1]["steady_inf_per_s"] - 1
            spread = best.get("steady_inf_per_s_spread")
            lines += [f"Next best `{aggs[1]['config']}` is {gain:.0%} slower"
                      + (f"; run-to-run spread on the winner is {spread:.0%}, so gaps smaller "
                         f"than that are noise." if spread else "; add REPEATS≥2 to see run-to-run noise.")]
        if best.get("gpu_share_of_work") is not None:
            lines += [f"GPU inference is {best['gpu_share_of_work']:.0%} of per-sample time; "
                      f"the rest is CPU preprocessing and scoring."]
        lines += [""]
    lines += ["| " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * len(COLS)]
    for a in aggs:
        lines.append("| " + " | ".join(fmt(a.get(k), nd) for k, _, nd in COLS) + " |")
    lines += ["", "## Runs", "",
              "| Run | Exit | Wall s | Fixed s | Prep s | Infer s | Score s | Overhead s |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['run_id']} | {r['exit_code']} | {fmt(r['wall_s'], 1)} | {fmt(r['fixed_s'], 1)} | "
                     f"{fmt(r['prep_s'], 1)} | {fmt(r['infer_s'], 1)} | {fmt(r['score_s'], 1)} | {fmt(r['overhead_s'], 1)} |")
    path.write_text("\n".join(lines) + "\n")
    return lines


def svg_bars(title, items, unit, nd=2):
    items = [(k, v) for k, v in items if v is not None]
    if not items:
        return ""
    w, bh, gap, left = 640, 26, 8, 190
    h = len(items) * (bh + gap) + 40
    vmax = max(v for _, v in items) or 1
    out = [f'<figure><figcaption>{html.escape(title)}</figcaption>'
           f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for i, (k, v) in enumerate(items):
        y = 10 + i * (bh + gap)
        bw = (w - left - 90) * v / vmax
        out.append(f'<text x="{left - 8}" y="{y + bh * 0.68}" text-anchor="end" class="lbl">{html.escape(k)}</text>'
                   f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bh}" rx="3" class="bar"/>'
                   f'<text x="{left + bw + 6}" y="{y + bh * 0.68}" class="val">{v:,.{nd}f} {unit}</text>')
    out.append("</svg></figure>")
    return "".join(out)


def write_html(path, md_lines, aggs):
    charts = [
        svg_bars("Steady-state inferences per second (higher is better)",
                 [(a["config"], a["steady_inf_per_s"]) for a in aggs], "inf/s"),
        svg_bars("Projected hours for the target run (lower is better)",
                 [(a["config"], a.get("proj_hours")) for a in aggs], "h", 1),
        svg_bars("Cost per audio hour (lower is better)",
                 [(a["config"], a.get("usd_per_audio_hour")) for a in aggs], "$", 4),
        svg_bars("GPU share of per-sample time",
                 [(a["config"], a["gpu_share_of_work"] * 100 if a["gpu_share_of_work"] is not None else None)
                  for a in aggs], "%", 0),
        svg_bars("Mean GPU utilisation", [(a["config"], a["gpu_util_mean"]) for a in aggs], "%", 0),
        svg_bars("Peak VRAM", [(a["config"], a["gpu_mem_peak_mib"]) for a in aggs], "MiB", 0),
    ]
    body, in_table = [], False
    for ln in md_lines:
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if set(ln) <= set("|-"):
                continue
            if not in_table:
                body.append("<div class=tw><table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table></div>")
            in_table = False
        if ln.startswith("## "):
            body.append(f"<h2>{html.escape(ln[3:])}</h2>")
        elif ln.startswith("# "):
            body.append(f"<h1>{html.escape(ln[2:])}</h1>")
        elif ln:
            t = html.escape(ln).replace("**", "")
            body.append(f"<p>{t}</p>")
    if in_table:
        body.append("</table></div>")
    # charts go after the summary paragraphs, before the run table
    idx = next((i for i, b in enumerate(body) if b.startswith("<h2>")), len(body))
    body[idx:idx] = ["<div class=charts>" + "".join(charts) + "</div>"]
    css = """:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--bar:#3b6fd8;--line:#e3e3e8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;--bar:#6f9bff;--line:#2c2c32}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1100px}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px}th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
figure{margin:0}figcaption{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto}.bar{fill:var(--bar)}
.lbl,.val{fill:var(--fg);font-size:13px}.val{fill:var(--muted)}"""
    path.write_text(f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                    f"<title>ASR sweep report</title><style>{css}</style>" + "\n".join(body))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--price-per-hour", type=float, default=0.0,
                    help="instance $/hour (check Lambda's current A10 price)")
    ap.add_argument("--target-inferences", type=int, default=44000,
                    help="size of the big run, in transcriptions (default: 1000 samples x 22 x 2 models)")
    args = ap.parse_args()

    sweep_path = args.sweep_dir / "sweep.json"
    sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}
    runs = [analyse_run(d, args.price_per_hour) for d in sorted(args.sweep_dir.iterdir())
            if (d / "run.json").exists()]
    if not runs:
        raise SystemExit(f"no runs under {args.sweep_dir}")
    aggs = aggregate(runs, args.price_per_hour, args.target_inferences)

    keys = sorted({k for a in aggs for k in a}, key=lambda k: (k != "config", k))
    with (args.sweep_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(aggs)
    with (args.sweep_dir / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)
    md = write_md(args.sweep_dir / "report.md", sweep, aggs, runs, args)
    write_html(args.sweep_dir / "report.html", md, aggs)
    print("\n".join(md[:12]))
    print(f"\nWrote summary.csv, runs.csv, report.md, report.html to {args.sweep_dir}")


if __name__ == "__main__":
    main()
