"""Upload a local folder to GCS and verify every file arrived with the same size.

Runs inside the benchmark image (it has google-cloud-storage), e.g.:
    python upload_results.py /upload lambda_bench_runs/<sweep_id>
Bucket and credentials come from GCS_BUCKET_NAME and GOOGLE_APPLICATION_CREDENTIALS.
Exits non-zero if anything is missing, so callers can refuse to terminate the instance.
"""
import os
import sys
from pathlib import Path

from google.cloud import storage


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: upload_results.py <local_dir> <gcs_prefix>")
    local_dir = Path(sys.argv[1])
    prefix = sys.argv[2].strip("/")
    bucket_name = os.environ["GCS_BUCKET_NAME"]

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        sys.exit(f"Nothing to upload in {local_dir}")

    bucket = storage.Client().bucket(bucket_name)
    expected = {}
    for p in files:
        key = f"{prefix}/{p.relative_to(local_dir).as_posix()}"
        bucket.blob(key).upload_from_filename(str(p))
        expected[key] = p.stat().st_size
        print(f"uploaded gs://{bucket_name}/{key}", flush=True)

    # Verify: every file present with the same size
    remote = {b.name: b.size for b in bucket.list_blobs(prefix=prefix + "/")}
    bad = [k for k, size in expected.items() if remote.get(k) != size]
    if bad:
        print("VERIFY FAILED for:", *bad, sep="\n  ", flush=True)
        sys.exit(1)
    print(f"VERIFIED {len(expected)} files in gs://{bucket_name}/{prefix}/", flush=True)


if __name__ == "__main__":
    main()
