"""Copy a run's final results pickle next to its logs in GCS, and verify the copy.

Runs inside the benchmark image (it has google-cloud-storage), where
scripts/cloud/run_and_terminate.sh mounts it at /app, e.g.:
    python copy_results.py results/transcribe_cache/<run_label>.pkl lambda_bench_runs/<sweep_id>/<run_id>
The copy is made inside GCS (no download) to <dest_prefix>/<file name>, so a run's results
and timing live in one folder and survive a cleanup of transcribe_cache/.
Bucket and credentials come from GCS_BUCKET_NAME and GOOGLE_APPLICATION_CREDENTIALS.
Exits non-zero if the source is missing or the copy's size differs.
"""
import os
import sys
from pathlib import PurePosixPath

from google.cloud import storage


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: copy_results.py <results_key> <dest_prefix>")
    key, prefix = sys.argv[1], sys.argv[2].strip("/")
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bucket = storage.Client().bucket(bucket_name)

    src = bucket.get_blob(key)
    if src is None:
        sys.exit(f"Not found: gs://{bucket_name}/{key}")
    dest_key = f"{prefix}/{PurePosixPath(key).name}"
    bucket.copy_blob(src, bucket, dest_key)

    dest = bucket.get_blob(dest_key)
    if dest is None or dest.size != src.size:
        sys.exit(f"VERIFY FAILED for gs://{bucket_name}/{dest_key}")
    print(f"VERIFIED copy gs://{bucket_name}/{dest_key} ({src.size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
