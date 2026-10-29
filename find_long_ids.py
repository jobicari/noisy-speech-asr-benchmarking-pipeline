"""Print the ids of samples that have at least one clip longer than a threshold.

Runs inside the benchmark image (it has google-cloud-storage), e.g.:
    python find_long_ids.py results/transcribe_cache/<run_label>.pkl [--min-duration 30]
Reads the results pickle from GCS (GCS_BUCKET_NAME) or, if the path exists, from local disk.
Prints the ids comma-separated on one line to stdout; everything else goes to stderr.
Rows without duration_s (written before it existed) are skipped.
"""
import argparse
import math
import os
import pickle
import sys
from pathlib import Path


def load_results(key):
    if Path(key).is_file():
        with open(key, "rb") as f:
            return pickle.load(f)
    from google.cloud import storage
    bucket = storage.Client().bucket(os.environ["GCS_BUCKET_NAME"])
    return pickle.loads(bucket.blob(key).download_as_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_key")
    parser.add_argument("--min-duration", type=float, default=30.0)
    args = parser.parse_args()

    rows = load_results(args.results_key)
    ids, missing = [], 0
    for row in rows:
        d = row.get("duration_s")
        if d is None or (isinstance(d, float) and math.isnan(d)):
            missing += 1
            continue
        if d > args.min_duration and row["idx"] not in ids:
            ids.append(row["idx"])

    print(f"{len(rows)} rows, {missing} without duration_s, "
          f"{len(ids)} samples with a clip > {args.min_duration} s", file=sys.stderr, flush=True)
    print(",".join(ids), flush=True)


if __name__ == "__main__":
    main()
