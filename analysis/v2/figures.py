#!/usr/bin/env python3
"""The README's figures, as PNG files in docs/figures/.

Run from the repo root in the project env, after downloading the v2 sweep folders into
results/lambda_bench_runs/ (see README.md in this folder):

    python analysis/v2/figures.py

Accuracy comes from the main run's results pickle, computed the same way as results.py (corpus
WER, and WER with each clip's errors capped at its reference length). Speed numbers are copied from
timing.md (computed by timing.py) and the tuning numbers from the "Tuning measurements" tables in
docs/getting-started.md.
"""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

OUT = Path("docs/figures")
V2 = Path("results/lambda_bench_runs/fullrun2000b/SAMPLES_PER_CALL_1_r1/"
          "fullRun__hf__986ea79c__2000samples__fullrun2000b_SAMPLES_PER_CALL_1_r1.pkl")

# Palette: chart chrome and categorical slots 1-2 of the reference palette, light mode.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PARAMS_LABEL = {"tiny": "39M", "base": "74M", "small": "244M", "medium": "769M", "large-v3": "1.55B",
                "turbo": "809M", "wav2vec2": "317M"}
PARAMS = {"tiny": 39e6, "base": 74e6, "small": 244e6, "medium": 769e6, "large-v3": 1.55e9,
          "turbo": 809e6, "wav2vec2": 317e6}
WHISPER = ["tiny", "base", "small", "medium", "large-v3"]
LABEL = {"tiny": "Whisper tiny", "base": "Whisper base", "small": "Whisper small",
         "medium": "Whisper medium", "large-v3": "Whisper large-v3", "turbo": "Whisper large-v3-turbo",
         "wav2vec2": "wav2vec2-large"}

# Inferences per second on one A10, main run (timing.md section 4).
INF_PER_S = {"tiny": 25.5, "base": 25.1, "small": 27.8, "medium": 15.7, "large-v3": 10.1,
             "turbo": 15.2, "wav2vec2": 107.6}
# GPU seconds per model, total and with each loop call cut to the model's loop-free mean
# (timing.md section 5: "without loops").
GPU_S = {"tiny": (1724, 677), "base": (1756, 794), "small": (1582, 1397), "medium": (2796, 2746),
         "large-v3": (4354, 4342), "turbo": (2886, 2872)}
# Wall time of the same 20-sample, 2-model run on the A10 as the pipeline was tuned
# (docs/getting-started.md, "Tuning measurements").
TUNING = [("Starting point:\none clip per GPU call", 274), ("All 22 corruptions\nin one call", 137),
          ("CPU worker pool\nfor corrupting audio", 91.6), ("Thread caps on\nthe workers", 78.6)]

CORR = {
    "none": "Clean audio", "codec_8k": "Opus codec, 8 kbps", "codec_6k": "Opus codec, 6 kbps",
    "PL_40_.1": "Packet loss, 10%", "PL_30_.2": "Packet loss, 20%", "PL_20_.3": "Packet loss, 30%",
    "PL_10_.4": "Packet loss, 40%", "PL_5_.5": "Packet loss, 50%",
    "clip_1": "Clipping, 10% of peak", "clip_05": "Clipping, 5% of peak", "clip_025": "Clipping, 2.5% of peak",
    "clipN_1": "Clipping, 10%, normalised", "clipN_05": "Clipping, 5%, normalised",
    "clipN_025": "Clipping, 2.5%, normalised",
    "SV_0.5": "Synthetic reverb", "RV_0.5_251": "Real room reverb", "SDO_8_100_15": "Dropouts (8 x 100 ms)",
    "WN_3snr": "White noise, 3 dB SNR", "BL_800_3000": "Band limit 800-3000 Hz",
    "BL_600_2000": "Band limit 600-2000 Hz", "BL_500_1000": "Band limit 500-1000 Hz",
    "WN_3-SV_.5-SDO_8_100_15": "Noise + reverb + dropouts",
}


def style():
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "text.color": INK,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 30, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    })


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def short(model):
    return (model.replace("openai_whisper-", "").replace("large-v3-turbo", "turbo")
            .replace("facebook_wav2vec2-large-960h-lv60-self", "wav2vec2"))


def load():
    """Per model: clean WER, WER over all corruptions; per corruption and model: capped WER."""
    sums = {}
    for r in pickle.load(open(V2, "rb")):
        err = r["w_subs"] + r["w_dels"] + r["w_ins"]
        k = (short(r["model"]), r["corr_type"])
        e, c, n = sums.get(k, (0, 0, 0))
        sums[k] = (e + err, c + min(err, r["num_words"]), n + r["num_words"])
    models = {m for m, _ in sums}
    acc = {}
    for m in models:
        e = sum(v[0] for (mm, _), v in sums.items() if mm == m)
        n = sum(v[2] for (mm, _), v in sums.items() if mm == m)
        acc[m] = {"Clean WER": 100 * sums[(m, "none")][0] / sums[(m, "none")][2], "All 22 corruptions": 100 * e / n,
                  "Params": PARAMS_LABEL[m]}
    capped = {}
    for (m, c), (_, cap, n) in sums.items():
        capped.setdefault(c, {})[m] = 100 * cap / n
    return acc, capped


def size_curve(acc):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = [PARAMS[m] for m in WHISPER]
    for col, color, name in (("All 22 corruptions", ORANGE, "Average over 22 corruptions"),
                             ("Clean WER", BLUE, "Clean audio")):
        y = [float(acc[m][col]) for m in WHISPER]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=f"Whisper, {name.lower()}",
                zorder=3)
        w = float(acc["wav2vec2"][col])
        ax.plot(PARAMS["wav2vec2"], w, marker="D", ms=9, color=color, mec=SURFACE, mew=2, ls="none", zorder=4)
    for m in WHISPER:
        ax.annotate(f"{float(acc[m]['All 22 corruptions']):.1f}%", (PARAMS[m], float(acc[m]["All 22 corruptions"])),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, color=INK2)
    w_all, w_clean = float(acc["wav2vec2"]["All 22 corruptions"]), float(acc["wav2vec2"]["Clean WER"])
    ax.annotate(f"wav2vec2: best on clean audio ({w_clean:.1f}%),\nbut {w_all:.1f}% under corruption",
                (PARAMS["wav2vec2"], w_all), textcoords="offset points", xytext=(14, 14), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks([PARAMS[m] for m in WHISPER],
                  [f"{m}\n{acc[m]['Params']}" for m in WHISPER])
    ax.minorticks_off()
    ax.set_xlim(28e6, 2.4e9)
    ax.set_ylim(0, 46)
    ax.set_xlabel("Whisper model and size (parameters, log scale)")
    ax.set_ylabel("Word error rate (%)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.set_title("Bigger Whisper models are much more robust to noisy audio")
    subtitle(ax, "Word error rate on 2,000 LibriSpeech clips, clean and averaged over 22 corruptions. "
                 "Diamonds: wav2vec2")
    ax.legend(loc="upper right")
    save(fig, "robustness_vs_size.png")


def heatmap(capped):
    models = ["tiny", "base", "small", "medium", "turbo", "large-v3", "wav2vec2"]
    rows = sorted(capped, key=lambda c: float(capped[c]["large-v3"]))
    vals = [[float(capped[c][m]) for m in models] for c in rows]
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    norm = LogNorm(vmin=1.5, vmax=100)
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if norm(v) > 0.45 else INK)
    heads = {"turbo": "Whisper\nlarge-v3-turbo", "wav2vec2": "wav2vec2\nlarge"}
    ax.set_xticks(range(len(models)), [heads.get(m, f"Whisper\n{m}") for m in models], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [CORR.get(c, c) for c in rows])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(models))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Word error rate (%) for every model and corruption", pad=62)
    ax.text(0, 1.07, "Sorted by Whisper large-v3. Errors per clip capped at the reference length, so a looping "
                      "transcript counts as 100%.", transform=ax.transAxes, color=INK2, fontsize=10.5)
    save(fig, "wer_by_corruption.png")


def accuracy_vs_speed(acc):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    offsets = {"tiny": (10, -4), "base": (10, -4), "small": (10, -4), "medium": (10, 2), "large-v3": (0, -20),
               "turbo": (10, -10), "wav2vec2": (-12, 8)}
    for m, s in INF_PER_S.items():
        y = float(acc[m]["All 22 corruptions"])
        color = ORANGE if m == "wav2vec2" else BLUE
        ax.plot(s, y, marker="D" if m == "wav2vec2" else "o", ms=10, color=color, mec=SURFACE, mew=2, zorder=3)
        dx, dy = offsets[m]
        ax.annotate(f"{LABEL[m]}  {y:.1f}%", (s, y), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else ("center" if m == "large-v3" else "right"), fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
    ax.set_xlim(7, 160)
    ax.set_ylim(0, 46)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Transcriptions per second on one A10 GPU (log scale)")
    ax.set_ylabel("Word error rate, 22 corruptions (%)")
    ax.set_title("Turbo is nearly as accurate as large-v3 at 1.5x the speed")
    subtitle(ax, "Accuracy against throughput. Lower and to the right is better")
    ax.grid(axis="x", visible=False)
    save(fig, "accuracy_vs_speed.png")


def loop_gpu_time():
    models = list(GPU_S)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    base = [GPU_S[m][1] / 60 for m in models]
    extra = [(GPU_S[m][0] - GPU_S[m][1]) / 60 for m in models]
    y = range(len(models))
    ax.barh(y, base, color=BLUE, height=0.62, label="Normal decoding", edgecolor=SURFACE, linewidth=2)
    ax.barh(y, extra, left=base, color=ORANGE, height=0.62, label="Extra time from repetition loops",
            edgecolor=SURFACE, linewidth=2)
    for i, m in enumerate(models):
        tot, nl = GPU_S[m]
        share = (tot - nl) / tot
        ax.text(tot / 60 + 1, i, f"{tot / 60:.0f} min" + (f"  ({share:.0%} of it from loops)" if share >= 0.1 else ""),
                va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(list(y), [LABEL[m].replace("Whisper ", "") for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("GPU time over the full run (minutes)")
    ax.set_xlim(0, 88)
    ax.grid(axis="y", visible=False)
    ax.set_title("Repetition loops make the smallest models as slow as larger ones")
    subtitle(ax, "On damaged audio tiny and base repeat a phrase until the token limit, "
                 "and the whole batch waits")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    save(fig, "loop_gpu_time.png")


def tuning():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = [n for n, _ in TUNING]
    vals = [v for _, v in TUNING]
    y = range(len(vals))
    ax.barh(y, vals, color=BLUE, height=0.6)
    for i, v in enumerate(vals):
        note = "" if i == 0 else f"   {vals[0] / v:.1f}x faster"
        ax.text(v + 3, i, f"{v:.0f} s{note}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 340)
    ax.set_xlabel("Wall time for the same 20-sample, 2-model run (seconds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Pipeline tuning made the benchmark 3.5x faster")
    subtitle(ax, "Each step added on top of the one before, measured on a Lambda A10")
    save(fig, "pipeline_speedups.png")


def main():
    style()
    acc, capped = load()
    size_curve(acc)
    heatmap(capped)
    accuracy_vs_speed(acc)
    loop_gpu_time()
    tuning()


if __name__ == "__main__":
    main()
