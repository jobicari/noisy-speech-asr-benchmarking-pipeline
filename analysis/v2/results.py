"""v2 (fullrun2000b) results: completeness, v1 agreement, WER tables, loops, fallback pass, long clips.

Run from the repo root with the project env:
    python analysis/v2/results.py
"""
import pickle
import zlib

import pandas as pd

R = "results/lambda_bench_runs/"
V2 = R + "fullrun2000b/SAMPLES_PER_CALL_1_r1/fullRun__hf__986ea79c__2000samples__fullrun2000b_SAMPLES_PER_CALL_1_r1.pkl"
V1 = "results/transcribe_cache/fullRun__hf__986ea79c__2000samples__fullrun2000.pkl"
LOOPS = R + "fullrun2000b_loops/WHISPER_FALLBACK_True_r1/fullRun__hf__a9498565__959samples__fullrun2000b_loops.pkl"
LONG2 = R + "fullrun2000b_longform/WHISPER_LONGFORM_sequential_r1/fullRun__hf__a9498565__8samples__fullrun2000b_longform.pkl"
LONG1 = "results/transcribe_cache/fullRun__hf__a9498565__8samples__fullrun2000_finish_longform.pkl"
ORDER = ["tiny", "base", "small", "medium", "large-v3", "turbo", "w2v2"]
KEY = ["idx", "model", "corr_type"]
COLS = ("idx", "model", "corr_type", "w_subs", "w_dels", "w_ins", "num_words",
        "c_subs", "c_dels", "c_ins", "num_chars", "hyp", "duration_s")


def md(df):
    df = df.reset_index()
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def ratio(text):
    """zlib compression ratio of the text, OpenAI's loop test (> 2.4)"""
    data = (text or "").encode()
    return len(data) / len(zlib.compress(data)) if data else 0.0


def load(path):
    rows = pickle.load(open(path, "rb"))
    df = pd.DataFrame([{k: r.get(k) for k in COLS}
                       | {"flag": r["sample_analysis"]["asr_flags"]["action_required"]} for r in rows])
    df["model"] = (df.model.str.replace("openai_whisper-", "", regex=False)
                   .str.replace("facebook_wav2vec2-large-960h-lv60-self", "w2v2", regex=False)
                   .str.replace("large-v3-turbo", "turbo", regex=False))
    df["err"] = df.w_subs + df.w_dels + df.w_ins
    df["cerr"] = df.c_subs + df.c_dels + df.c_ins
    df["hw"] = df.hyp.fillna("").str.split().str.len()
    df["cr"] = df.hyp.map(ratio)
    df["loop"] = (df.cr > 2.4) & ~df.model.eq("w2v2")
    return df


def wer(d):
    return 100 * d.err.sum() / d.num_words.sum()


def capped(d):
    return 100 * d.err.clip(upper=d.num_words).sum() / d.num_words.sum()


def cer(d):
    return 100 * d.cerr.sum() / d.num_chars.sum()


def by_model(df, fn):
    return df.groupby("model").apply(fn, include_groups=False).reindex(ORDER)


v2, v1 = load(V2), load(V1)

print("## Completeness")
print(len(v2), "rows,", v2[KEY].drop_duplicates().shape[0], "unique,", v2.idx.nunique(), "samples,",
      v2.corr_type.nunique(), "corruptions,", v2.model.nunique(), "models")
clean = v2[(v2.corr_type == "none") & (v2.model == "tiny")]
print("duration_s missing:", v2.duration_s.isna().sum(), "| samples with a clip > 30 s:", v2[v2.duration_s > 30].idx.nunique())
print("clean audio hours:", round(clean.duration_s.sum() / 3600, 2), "| reference words per corruption:", clean.num_words.sum())
print("empty hypotheses:", v2[v2.hw == 0].groupby("model").size().to_dict())

print("\n## v1 vs v2")
m = v1.merge(v2, on=KEY, suffixes=("_1", "_2"))
m["same"] = m.hyp_1 == m.hyp_2
g = m.groupby("model")
print(md(pd.DataFrame({
    "identical_hyp_pct": 100 * g.same.mean(),
    "wer_v1": g.apply(lambda d: 100 * d.err_1.sum() / d.num_words_1.sum(), include_groups=False),
    "wer_v2": g.apply(lambda d: 100 * d.err_2.sum() / d.num_words_2.sum(), include_groups=False),
}).reindex(ORDER).round(2)))
print("identical overall:", round(100 * m.same.mean(), 2), "%")
print("differing rows by corruption (top 8):", m[~m.same].corr_type.value_counts().head(8).to_dict())

print("\n## Main table")
t = pd.DataFrame({
    "clean": by_model(v2[v2.corr_type == "none"], wer), "all": by_model(v2, wer), "capped": by_model(v2, capped),
    "cer_all": by_model(v2, cer),
    "median_row_wer": by_model(v2, lambda d: 100 * (d.err / d.num_words).median()),
    "loop_rows": v2.groupby("model").loop.sum().reindex(ORDER),
    "empty": by_model(v2, lambda d: (d.hw == 0).sum()),
}).round(2)
print(md(t))
print("all / clean:", (t["all"] / t["clean"]).round(1).to_dict())
print("capped / clean:", (t["capped"] / t["clean"]).round(1).to_dict())

print("\n## Per corruption, corpus WER")
pc = v2.groupby(["corr_type", "model"]).apply(wer, include_groups=False).unstack()[ORDER].round(1).sort_values("large-v3")
print(md(pc))
print("best model per corruption:", pc.idxmin(axis=1).value_counts().to_dict())
print("\n## Per corruption, capped WER")
print(md(v2.groupby(["corr_type", "model"]).apply(capped, include_groups=False).unstack()[ORDER].round(1).loc[pc.index]))

print("\n## Loop rows by corruption")
print(md(v2[v2.loop].groupby(["corr_type", "model"]).size().unstack(fill_value=0)))

print("\n## Error mix (% of S, D, I)")
s = v2.groupby("model")[["w_subs", "w_dels", "w_ins"]].sum()
print(md((100 * s.div(s.sum(1), axis=0)).reindex(ORDER).round(1)))

print("\n## Capped WER by clip duration")
v2["dur"] = pd.cut(v2.duration_s, [0, 5, 10, 15, 20, 30, 40])
print(md(v2.groupby(["dur", "model"], observed=True).apply(capped, include_groups=False).unstack()[ORDER].round(1)))
print("rows per bucket:", v2[v2.model == "tiny"].groupby("dur", observed=True).size().to_dict())

print("\n## Audio-health flag (action_required) vs capped WER")
print(md(v2.groupby(["flag", "model"]).apply(capped, include_groups=False).unstack()[ORDER].round(1)))
print("flagged clips by corruption:", v2[(v2.model == "tiny") & v2.flag].corr_type.value_counts().to_dict())

print("\n## Looped-clip pass (fallback)")
lp = load(LOOPS)
print(len(lp), "rows,", lp[["idx", "corr_type"]].drop_duplicates().shape[0], "clips")
j = lp.merge(v2, on=KEY, suffixes=("_fb", "_raw"))


def side_by_side(df):
    g = df.groupby("model")
    f = lambda fn: g.apply(fn, include_groups=False)
    return pd.DataFrame({
        "rows": g.size(), "loops_raw": g.loop_raw.sum(), "loops_fb": g.loop_fb.sum(),
        "wer_raw": f(lambda d: 100 * d.err_raw.sum() / d.num_words_raw.sum()),
        "wer_fb": f(lambda d: 100 * d.err_fb.sum() / d.num_words_fb.sum()),
        "capped_raw": f(lambda d: 100 * d.err_raw.clip(upper=d.num_words_raw).sum() / d.num_words_raw.sum()),
        "capped_fb": f(lambda d: 100 * d.err_fb.clip(upper=d.num_words_fb).sum() / d.num_words_fb.sum()),
        "fb_better": f(lambda d: (d.err_fb < d.err_raw).sum()),
        "fb_worse": f(lambda d: (d.err_fb > d.err_raw).sum()),
        "same_hyp": f(lambda d: (d.hyp_fb == d.hyp_raw).sum()),
        "empty_fb": f(lambda d: (d.hw_fb == 0).sum()),
    }).reindex(ORDER[:-1]).round(1)


print("All rerun clips:")
print(md(side_by_side(j)))
print("Only rows that looped in the main run:")
print(md(side_by_side(j[j.loop_raw])))
sub = v2.set_index(KEY).copy()
fb = lp.set_index(KEY)
sub.loc[fb.index, ["err", "num_words"]] = fb[["err", "num_words"]]
sub = sub.reset_index()
print("whole-run WER with the fallback rows substituted:", by_model(sub, wer).round(2).to_dict())
print("BL_500_1000 with fallback:", by_model(sub[sub.corr_type == "BL_500_1000"], wer).round(1).to_dict())
print("combined corruption with fallback:", by_model(sub[sub.corr_type == "WN_3-SV_.5-SDO_8_100_15"], wer).round(1).to_dict())

print("\n## Long clips (8 samples with a clip > 30 s)")
lf, lf1 = load(LONG2), load(LONG1)
ids = lf.idx.unique()
ch = v2[v2.idx.isin(ids) & ~v2.model.eq("w2v2")]
print(md(pd.DataFrame({
    "chunked_v2": by_model(ch, wer), "sequential_v2": by_model(lf, wer), "sequential_v1": by_model(lf1, wer),
    "chunked_v2_clips_over_30s": by_model(ch[ch.duration_s > 30], wer),
    "sequential_v2_clips_over_30s": by_model(lf[lf.duration_s > 30], wer),
}).reindex(ORDER[:-1]).round(1)))
print(len(lf), "rows,", (lf.duration_s > 30).sum(), "rows over 30 s")
