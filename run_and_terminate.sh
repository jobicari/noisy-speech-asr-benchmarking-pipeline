#!/usr/bin/env bash
# Unattended run on Lambda: sweep -> analyze -> upload the sweep folder to GCS
# -> verify -> terminate this instance (stops billing).
#
# Takes the same env vars as run_sweep.sh, plus:
#   SWEEP_ID            required, also the GCS folder name
#   LAMBDA_API_KEY_FILE file holding a Lambda Cloud API key (default ~/lambda-api-key, chmod 600)
#   LAMBDA_INSTANCE_ID  optional; found automatically from this machine's public IP
#   TERMINATE           1 (default) terminate at the end, 0 to only upload
#   TERMINATE_DELAY     seconds to wait before terminating, so you can Ctrl-C (default 120)
#   PRICE_PER_HOUR      for analyze.py (default 1.29)
#   TARGET_INFERENCES   for analyze.py (default: unset, analyze.py's own default)
#   GCS_PREFIX          bucket folder for sweeps (default lambda_bench_runs)
#
# The instance is terminated only if the upload is verified. Terminating deletes
# everything on the instance, so a failed upload leaves it running for you to fix.
set -euo pipefail

: "${SWEEP_ID:?Set SWEEP_ID (it is also the GCS folder name)}"
IMAGE="${IMAGE:-asr-benchmark:lambda}"
ENV_FILE="${ENV_FILE:-$HOME/noisy-speech-asr-benchmarking-pipeline/.env.docker}"
KEY_FILE="${KEY_FILE:-$HOME/gcs-key.json}"
OUT_ROOT="${OUT_ROOT:-./results/lambda_bench_runs}"
LAMBDA_API_KEY_FILE="${LAMBDA_API_KEY_FILE:-$HOME/lambda-api-key}"
LAMBDA_API="${LAMBDA_API:-https://cloud.lambdalabs.com/api/v1}"
TERMINATE="${TERMINATE:-1}"
TERMINATE_DELAY="${TERMINATE_DELAY:-120}"
PRICE_PER_HOUR="${PRICE_PER_HOUR:-1.29}"
GCS_PREFIX="${GCS_PREFIX:-lambda_bench_runs}"
export IMAGE ENV_FILE KEY_FILE OUT_ROOT SWEEP_ID

SWEEP_DIR="$OUT_ROOT/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"
# Log next to (not inside) the sweep folder, so the upload never sees a file that is still growing
exec > >(tee -a "$OUT_ROOT/$SWEEP_ID.pipeline.log") 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# ---- Check termination prerequisites BEFORE the long run ---------------------
if [[ "$TERMINATE" == "1" ]]; then
  [[ -r "$LAMBDA_API_KEY_FILE" ]] || { log "No API key at $LAMBDA_API_KEY_FILE (or set TERMINATE=0)"; exit 1; }
  API_KEY=$(tr -d '[:space:]' < "$LAMBDA_API_KEY_FILE")
  if [[ -z "${LAMBDA_INSTANCE_ID:-}" ]]; then
    my_ip=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
    LAMBDA_INSTANCE_ID=$(curl -fsS -u "$API_KEY:" "$LAMBDA_API/instances" | python3 -c '
import json, sys
ip = sys.argv[1]
ids = [i["id"] for i in json.load(sys.stdin).get("data", []) if i.get("ip") == ip]
print(ids[0] if len(ids) == 1 else "")' "$my_ip")
    [[ -n "$LAMBDA_INSTANCE_ID" ]] || { log "Could not find this instance ($my_ip) in the Lambda API. Set LAMBDA_INSTANCE_ID."; exit 1; }
  fi
  log "Will terminate instance $LAMBDA_INSTANCE_ID after a verified upload"
else
  log "TERMINATE=0: will upload but not terminate"
fi

# ---- 1. Sweep ----------------------------------------------------------------
log "Starting sweep $SWEEP_ID"
./run_sweep.sh || log "run_sweep.sh exited with $? (uploading anyway)"

# ---- 2. Analyze (never blocks the upload) ------------------------------------
analyze_args=(--price-per-hour "$PRICE_PER_HOUR")
[[ -n "${TARGET_INFERENCES:-}" ]] && analyze_args+=(--target-inferences "$TARGET_INFERENCES")
python3 analyze.py "$SWEEP_DIR" "${analyze_args[@]}" || log "analyze.py failed (uploading anyway)"

# ---- 3. Upload + verify ------------------------------------------------------
log "Uploading $SWEEP_DIR to $GCS_PREFIX/$SWEEP_ID"
abs_sweep=$(cd "$SWEEP_DIR" && pwd)
if ! $DOCKER run --rm --env-file "$ENV_FILE" \
      -v "$KEY_FILE:/secrets/key.json:ro" \
      -v "$abs_sweep:/upload:ro" \
      -v "$PWD/upload_results.py:/app/upload_results.py:ro" \
      --entrypoint python "$IMAGE" /app/upload_results.py /upload "$GCS_PREFIX/$SWEEP_ID"; then
  log "UPLOAD NOT VERIFIED. Instance left running so nothing is lost."
  exit 1
fi

# ---- 4. Terminate ------------------------------------------------------------
[[ "$TERMINATE" == "1" ]] || { log "Done (TERMINATE=0)."; exit 0; }
log "Terminating $LAMBDA_INSTANCE_ID in ${TERMINATE_DELAY}s. Ctrl-C to cancel."
sleep "$TERMINATE_DELAY"
curl -fsS -u "$API_KEY:" -H "Content-Type: application/json" \
  -d "{\"instance_ids\": [\"$LAMBDA_INSTANCE_ID\"]}" \
  "$LAMBDA_API/instance-operations/terminate"
log "Terminate request sent."
