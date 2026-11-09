#!/usr/bin/env python3
"""The repository's social preview image (docs/figures/social_preview.png, 1280x640).

Upload it under Settings > General > Social preview, so links to the repo (LinkedIn,
Slack, X) show it. The heatmap is the capped WER from docs/demo/data.json.

    python analysis/v2/social_preview.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
RAMP = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NAMES = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium",
         "turbo": "turbo", "large-v3": "large-v3", "wav2vec2": "wav2vec2"}

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"

heat = json.loads(Path("docs/demo/data.json").read_text())["heatmap"]
vals = heat["capped"]

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=SURFACE)
fig.text(0.06, 0.82, "How speech recognition\nbreaks on bad audio", fontsize=36, fontweight="bold",
         color=INK, va="top", linespacing=1.1)
fig.text(0.06, 0.54, "7 models  ·  22 kinds of audio damage\n2,000 recordings  ·  308,000 transcriptions",
         fontsize=19, color=INK2, va="top", linespacing=1.5)
fig.text(0.06, 0.33, "Whisper tiny to large-v3, and wav2vec2.\nNoise, reverb, codecs, packet loss,\nclipping, band limiting.",
         fontsize=15, color=MUTED, va="top", linespacing=1.5)
fig.text(0.06, 0.06, "github.com/jobicari/noisy-speech-asr-benchmarking-pipeline", fontsize=13, color=MUTED)

ax = fig.add_axes([0.63, 0.10, 0.32, 0.74])
cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
ax.imshow(vals, aspect="auto", cmap=cmap, norm=LogNorm(vmin=1.5, vmax=100))
ax.set_xticks(range(len(heat["models"])), [NAMES[m] for m in heat["models"]], rotation=45, ha="left",
              fontsize=11, color=INK2)
ax.xaxis.tick_top()
ax.set_yticks([])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_ylabel("clean audio  →  hardest damage", fontsize=12, color=MUTED, labelpad=8)
fig.text(0.79, 0.04, "Word error rate: light = few errors, dark = most words wrong", fontsize=10.5,
         color=MUTED, ha="center")

out = Path("docs/figures/social_preview.png")
fig.savefig(out, facecolor=SURFACE)
print(out, fig.get_size_inches() * fig.dpi)
