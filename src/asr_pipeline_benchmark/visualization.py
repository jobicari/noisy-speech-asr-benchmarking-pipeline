import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from asr_pipeline_benchmark.configuration import Config
config = Config()


def plot_feature_scatter_grid(
    df,
    features,
    target="wer",
    ncols=4,
    figsize_per_cell=(3.6, 3.0),
    alpha=0.4,
    add_trend=True,
    share_target_axis=True,
    title=None,
    save_path=None,
):
    """
    Automatically build a scatter-plot grid of `target` (WER) against every
    pruned feature.

    - `features` can be the raw `pruned_features` list (the target is filtered
      out automatically, so you can pass it as-is).
    - Grid dimensions are inferred from the number of valid features.
    - Non-numeric / missing columns are silently skipped.
    """
    # De-duplicate, drop the target, keep only columns that exist & are numeric
    feats = [
        c for c in dict.fromkeys(features)
        if c != target and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    n = len(feats)
    if n == 0:
        raise ValueError("No numeric features to plot after filtering.")

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Shared y-limits keep the WER axis comparable panel-to-panel
    if share_target_axis:
        y_min, y_max = df[target].min(), df[target].max()

    for ax, col in zip(axes_flat, feats):
        data = df[[col, target]].dropna()

        ax.scatter(data[col], data[target], alpha=alpha, s=15,
                   edgecolors="w", linewidths=0.3)

        # Spearman rho: monotonic strength, robust to the non-linear WER spread
        rho = data[col].corr(data[target], method="spearman")

        if add_trend and len(data) > 1:
            m, b = np.polyfit(data[col], data[target], 1)
            xs = np.linspace(data[col].min(), data[col].max(), 100)
            ax.plot(xs, m * xs + b, color="crimson", lw=1.5, alpha=0.8)

        ax.set_title(f"{col}\nρ = {rho:.2f}", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel(target, fontsize=8)
        ax.tick_params(labelsize=7)

        if share_target_axis:
            ax.set_ylim(y_min, y_max)

    # Hide any unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title or f"WER vs pruned features", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, axes

