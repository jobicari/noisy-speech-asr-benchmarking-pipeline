"""Print the clips where a Whisper model went into a repetition loop.

Runs inside the benchmark image (it has google-cloud-storage), e.g.:
    python find_loop_clips.py results/transcribe_cache/<run_label>.pkl [--min-ratio 2.4]
Reads the results pickle from GCS (GCS_BUCKET_NAME) or, if the path exists, from local disk.
A transcript counts as a loop when its text compresses more than --min-ratio times with
zlib, the test OpenAI's Whisper uses to trigger temperature fallback. It is computed on the
normalized hypothesis, which is what the results store.
Prints "sample_id|corr_type" pairs comma-separated on one line to stdout (the CLIPS format);
everything else goes to stderr.
"""
import argparse
import sys
import zlib

from find_long_ids import load_results


def compression_ratio(text):
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_key")
    parser.add_argument("--min-ratio", type=float, default=2.4)
    args = parser.parse_args()

    rows = load_results(args.results_key)
    clips, n_loops = [], 0
    for row in rows:
        if "whisper" not in row["model"].lower():
            continue
        if compression_ratio(row["hyp"]) > args.min_ratio:
            n_loops += 1
            clip = f"{row['idx']}|{row['corr_type']}"
            if clip not in clips:
                clips.append(clip)

    n_samples = len({c.split("|")[0] for c in clips})
    print(f"{len(rows)} rows, {n_loops} looped Whisper transcripts, "
          f"{len(clips)} clips in {n_samples} samples", file=sys.stderr, flush=True)
    print(",".join(clips), flush=True)


if __name__ == "__main__":
    main()
