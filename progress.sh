#!/usr/bin/env bash
# Show the progress of the run that is currently writing to a container.log.
#
# Usage:
#   ./progress.sh                       # newest container.log under ./results/lambda_bench_runs
#   ./progress.sh results/lambda_bench_runs/fullrun1      # newest log in that sweep
#   ./progress.sh path/to/container.log
#   watch -n 30 ./progress.sh           # refresh every 30 s (in a second tmux pane)
set -euo pipefail

target="${1:-${OUT_ROOT:-./results/lambda_bench_runs}}"
if [[ -f "$target" ]]; then
  log="$target"
else
  log=$(find "$target" -name container.log -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi
[[ -n "${log:-}" && -f "$log" ]] || { echo "No container.log found under $target"; exit 1; }

python3 - "$log" <<'PY'
import json, re, sys
from datetime import datetime, timezone

log = sys.argv[1]
n_total = models = resumed = 0
checkpoint = None
first_ts = last_ts = None
samples = set()
done = cached = False

for line in open(log, errors="replace"):
    if line.startswith("[TIMING] "):
        try:
            r = json.loads(line[9:])
        except ValueError:
            continue
        ts = datetime.fromisoformat(r["ts"])
        if r.get("event") == "sample":
            samples.add(r["sample_idx"])
            first_ts = first_ts or ts
            last_ts = ts
        continue
    if m := re.search(r"Run Batch Checkpoint Inference on (\d+) samples", line):
        n_total = int(m.group(1))
    elif line.startswith("Initializing "):
        models += 1
    elif m := re.search(r"Resuming pipeline from checkpoint \d+ \(sample (\d+)\)", line):
        resumed = int(m.group(1))
    elif m := re.search(r"=== Checkpoint #(\d+) ===", line):
        checkpoint = int(m.group(1))
    elif "Loaded test results from cache" in line:
        cached = True
    elif "RUN COMPLETED SUCCESSFULLY" in line:
        done = True

print(f"log:        {log}")
if cached:
    print("status:     loaded cached results, no inference ran (RUN_POSTFIX reused?)")
    sys.exit()
if not n_total:
    print("status:     starting up (dataset / models loading)")
    sys.exit()

done_n = resumed + len(samples)
pct = 100 * done_n / n_total
print(f"models:     {models}")
print(f"checkpoint: {checkpoint if checkpoint is not None else '-'}"
      + (f"  (resumed at sample {resumed})" if resumed else ""))
print(f"samples:    {done_n:,}/{n_total:,}  ({pct:.1f}%)")

def hm(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"

if first_ts and len(samples) > 1:
    elapsed = (last_ts - first_ts).total_seconds()
    rate = (len(samples) - 1) / elapsed if elapsed > 0 else 0   # samples/s since the first one finished
    since = (datetime.now(timezone.utc) - last_ts).total_seconds()
    eta = (n_total - done_n) / rate if rate else 0
    print(f"elapsed:    {hm(elapsed)}   rate: {rate * 60:.1f} samples/min")
    print(f"ETA:        {hm(eta)}" + ("" if done else f"   (last sample record {int(since)} s ago)"))
print("status:     " + ("COMPLETED" if done else "running"))
PY
