#!/usr/bin/env bash
# Unattended run on Lambda: sweep -> check it succeeded -> long-clip side run
# -> analyze -> upload to GCS -> verify -> terminate this instance (stops billing).
#
# Takes the same env vars as run_sweep.sh, plus:
#   SWEEP_ID             required, also the GCS folder name
#   LAMBDA_API_KEY_FILE  file holding a Lambda Cloud API key (default ~/lambda-api-key, chmod 600)
#   LAMBDA_INSTANCE_ID   optional; found automatically from this machine's public IP
#   TERMINATE            1 (default) terminate at the end, 0 to only upload
#   TERMINATE_DELAY      seconds to wait before terminating, so you can cancel (default 120)
#   TERMINATE_ON_FAILURE 0 (default) keep the instance up if the main sweep failed, 1 terminate anyway
#   PRICE_PER_HOUR       for analyze.py (default 1.29)
#   TARGET_INFERENCES    for analyze.py (default: unset, analyze.py's own default)
#   GCS_PREFIX           bucket folder for sweeps (default lambda_bench_runs)
#   LONGFORM_RUN         1 (default) after the main sweep, rerun every sample with a clip
#                        > LONGFORM_MIN_S seconds with WHISPER_LONGFORM=sequential; 0 to skip
#   LONGFORM_MIN_S       duration threshold for the side run (default 30)
#   LONGFORM_MODELS      models for the side run (default: the Whisper models in HF_MODELS)
#   LOOP_RUN             1 (default) after the main sweep, rerun every clip where a Whisper model
#                        looped (find_loop_clips.py) with WHISPER_FALLBACK=True; 0 to skip
#   LOOP_MIN_RATIO       compression ratio above which a transcript counts as a loop (default 2.4)
#   LOOP_MODELS          models for the loop side run (default: the Whisper models in HF_MODELS)
#   NTFY_TOPIC           optional; push notifications to https://ntfy.sh/<topic> (keep it private).
#                        Read from ENV_FILE if not set in the shell.
#   STALL_MIN            with NTFY_TOPIC, alert when no run log has grown for this many minutes
#                        (default 30; 0 turns it off). Alert only: the run is not stopped.
#   DETACH               1 (default) re-run in the background, detached from the terminal, so a
#                        dropped SSH session can't kill it; 0 to stay in the foreground
#
# The instance is terminated only if the main sweep succeeded and the upload is
# verified. A failed side run is uploaded and reported but does not block termination.
#
# Run it from anywhere, e.g. ./scripts/cloud/run_and_terminate.sh: it works from the repo root,
# so relative paths (OUT_ROOT, ENV_FILE, KEY_FILE) are relative to the repo root.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# Helper scripts mounted into the image at run time (the image only holds app.py and src/)
CONTAINER_SCRIPTS="$REPO_ROOT/scripts/container"
cd "$REPO_ROOT"

: "${SWEEP_ID:?Set SWEEP_ID (it is also the GCS folder name)}"
IMAGE="${IMAGE:-asr-benchmark:lambda}"
ENV_FILE="${ENV_FILE:-$HOME/noisy-speech-asr-benchmarking-pipeline/.env.docker}"
KEY_FILE="${KEY_FILE:-$HOME/gcs-key.json}"
OUT_ROOT="${OUT_ROOT:-./results/lambda_bench_runs}"
LAMBDA_API_KEY_FILE="${LAMBDA_API_KEY_FILE:-$HOME/lambda-api-key}"
LAMBDA_API="${LAMBDA_API:-https://cloud.lambdalabs.com/api/v1}"
TERMINATE="${TERMINATE:-1}"
TERMINATE_DELAY="${TERMINATE_DELAY:-120}"
TERMINATE_ON_FAILURE="${TERMINATE_ON_FAILURE:-0}"
PRICE_PER_HOUR="${PRICE_PER_HOUR:-1.29}"
GCS_PREFIX="${GCS_PREFIX:-lambda_bench_runs}"
LONGFORM_RUN="${LONGFORM_RUN:-1}"
LONGFORM_MIN_S="${LONGFORM_MIN_S:-30}"
STALL_MIN="${STALL_MIN:-30}"
LOOP_RUN="${LOOP_RUN:-1}"
LOOP_MIN_RATIO="${LOOP_MIN_RATIO:-2.4}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
# Fall back to NTFY_TOPIC in the env file
[[ -n "$NTFY_TOPIC" ]] || NTFY_TOPIC=$(grep -E '^NTFY_TOPIC=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)
export IMAGE ENV_FILE KEY_FILE OUT_ROOT SWEEP_ID

SWEEP_DIR="$OUT_ROOT/$SWEEP_ID"
SIDE_ID="${SWEEP_ID}_longform"
SIDE_DIR="$OUT_ROOT/$SIDE_ID"
LOOP_ID="${SWEEP_ID}_loops"
LOOP_DIR="$OUT_ROOT/$LOOP_ID"
PIPELINE_LOG="$OUT_ROOT/$SWEEP_ID.pipeline.log"
mkdir -p "$SWEEP_DIR"

# A closed SSH session sends SIGHUP to everything started from it. Re-run in a new
# session (setsid) with no terminal, so the run survives it even outside tmux.
if [[ "${DETACH:-1}" == "1" && -z "${_RT_DETACHED:-}" ]]; then
  _RT_DETACHED=1 setsid nohup "$SCRIPT_DIR/run_and_terminate.sh" "$@" >/dev/null 2>&1 </dev/null &
  echo "Running detached (pid $!). Follow:  tail -f $PIPELINE_LOG"
  echo "Stop:  pkill -f run_and_terminate.sh  (a running container keeps going: docker stop it)"
  exit 0
fi
# Log next to (not inside) the sweep folder, so the upload never sees a file that is still growing
exec > >(tee -a "$PIPELINE_LOG") 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }

# Push a notification if NTFY_TOPIC is set. Never fails the script.
notify() {
  [[ -n "$NTFY_TOPIC" ]] || return 0
  curl -fsS -m 15 -H "Title: $SWEEP_ID: $1" -d "$2" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 \
    || log "Notification failed (ignored)"
}
# Any unexpected exit (set -e, a failed command) also sends a notification
NOTIFIED_EXIT=0   # set to 1 where an exit already sent its own notification
WATCH_PID=
# Stop the stall watcher and its sleep, which would otherwise hold the log open for a minute
trap 'rc=$?; [[ -n "$WATCH_PID" ]] && kill $(pgrep -P "$WATCH_PID") "$WATCH_PID" 2>/dev/null; [[ $rc -ne 0 && $NOTIFIED_EXIT == 0 ]] && notify "script exited ($rc)" "run_and_terminate.sh stopped with exit $rc. Instance left running. See $PIPELINE_LOG"' EXIT

# Stall alert: a hung run (e.g. a killed prep worker blocks the pool forever) keeps
# billing without failing. Alert once per stall when the newest container.log of this
# sweep and its side runs has not grown for STALL_MIN minutes; alert again after it moves.
watch_stall() {
  local alerted="" log age
  while sleep 60; do
    log=$(ls -t "$OUT_ROOT/$SWEEP_ID"*/*/container.log 2>/dev/null | head -1 || true)
    [[ -n "$log" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if (( age < STALL_MIN * 60 )); then
      alerted=""
    elif [[ "$alerted" != "$log" ]]; then
      log "No log output for $((age / 60)) min in $log"
      notify "no progress for $((age / 60)) min" "Nothing new in $log for $((age / 60)) min. The run may be stuck; the instance is still billing."
      alerted=$log
    fi
  done
}

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# A sweep succeeded if every run exited 0 and its log says RUN COMPLETED SUCCESSFULLY
sweep_ok() {
  local dir=$1 n=0 run
  for run in "$dir"/*/; do
    [[ -f "$run/run.json" ]] || continue
    n=$((n + 1))
    grep -q '"exit_code": 0,' "$run/run.json" || { log "FAILED: $run (non-zero exit)"; return 1; }
    grep -q 'RUN COMPLETED SUCCESSFULLY' "$run/container.log" || { log "FAILED: $run (no success line)"; return 1; }
  done
  [[ $n -gt 0 ]] || { log "FAILED: no runs in $dir"; return 1; }
}

# Upload a sweep folder to GCS and verify it
upload() {
  local dir=$1 dest=$2 abs
  log "Uploading $dir to $GCS_PREFIX/$dest"
  abs=$(cd "$dir" && pwd)
  $DOCKER run --rm --env-file "$ENV_FILE" \
    -v "$KEY_FILE:/secrets/key.json:ro" \
    -v "$abs:/upload:ro" \
    -v "$CONTAINER_SCRIPTS/upload_results.py:/app/upload_results.py:ro" \
    --entrypoint python "$IMAGE" /app/upload_results.py /upload "$GCS_PREFIX/$dest"
}

# Copy each run's final results pickle (named in its container.log) into the sweep's
# GCS folder, next to that run's logs
copy_results() {
  local dir=$1 dest=$2 run key rc=0
  for run in "$dir"/*/; do
    [[ -f "$run/container.log" ]] || continue
    key=$(grep -A1 -E '(dumped|Loaded) test results (to|from) cache:' "$run/container.log" | tail -1 | tr -d '[:space:]' || true)
    [[ -n "$key" ]] || { log "No results key in $run/container.log: nothing to copy"; rc=1; continue; }
    $DOCKER run --rm --env-file "$ENV_FILE" \
      -v "$KEY_FILE:/secrets/key.json:ro" \
      -v "$CONTAINER_SCRIPTS/copy_results.py:/app/copy_results.py:ro" \
      --entrypoint python "$IMAGE" /app/copy_results.py "$key" "$GCS_PREFIX/$dest/$(basename "$run")" || rc=1
  done
  return $rc
}

analyze() {
  python3 "$SCRIPT_DIR/analyze.py" "$1" --price-per-hour "$PRICE_PER_HOUR" "${@:2}" || log "analyze.py failed on $1 (uploading anyway)"
}

# Run a finder script (find_long_ids.py, find_loop_clips.py) in the image on the results
# pickle of every main run, and print the union of what they found, comma-separated
# (its log lines go to stderr, so they don't end up in the result)
find_in_main_results() {
  local script=$1 out="" clog key found
  shift
  for clog in "$SWEEP_DIR"/*/container.log; do
    key=$(grep -A1 -E '(dumped|Loaded) test results (to|from) cache:' "$clog" | tail -1 | tr -d '[:space:]' || true)
    [[ -n "$key" ]] || { log "No results key in $clog" >&2; continue; }
    found=$($DOCKER run --rm --env-file "$ENV_FILE" \
      -v "$KEY_FILE:/secrets/key.json:ro" \
      -v "$CONTAINER_SCRIPTS/find_long_ids.py:/app/find_long_ids.py:ro" \
      -v "$CONTAINER_SCRIPTS/$script:/app/$script:ro" \
      --entrypoint python "$IMAGE" "/app/$script" "$key" "$@") \
      || { log "$script failed on $key" >&2; continue; }
    out="${out:+$out,}$found"
  done
  tr ',' '\n' <<< "$out" | sed '/^$/d' | sort -u | paste -sd, -
}

# The Whisper models in HF_MODELS: wav2vec2 has no chunking and no decoding loop
whisper_models() {
  grep -E '^HF_MODELS=' "$ENV_FILE" | cut -d= -f2- | tr -d '"'"'"' ' | tr ',' '\n' | { grep -i whisper || true; } | paste -sd, -
}

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
  log "Will terminate instance $LAMBDA_INSTANCE_ID after a successful sweep and a verified upload"
else
  log "TERMINATE=0: will upload but not terminate"
fi
notify "started" "Sweep $SWEEP_ID started on $(hostname)"
if [[ -n "$NTFY_TOPIC" && "$STALL_MIN" -gt 0 ]]; then
  watch_stall &
  WATCH_PID=$!
fi

# ---- 1. Main sweep -----------------------------------------------------------
log "Starting sweep $SWEEP_ID"
"$SCRIPT_DIR/run_sweep.sh" || log "run_sweep.sh exited with $?"
main_ok=1
sweep_ok "$SWEEP_DIR" || main_ok=0
if [[ $main_ok == 1 ]]; then
  log "Main sweep succeeded"
else
  notify "MAIN RUN FAILED" "Sweep $SWEEP_ID failed. Uploading logs. $( [[ $TERMINATE_ON_FAILURE == 1 ]] && echo 'Instance will terminate.' || echo 'Instance left running.')"
fi

analyze_args=()
[[ -n "${TARGET_INFERENCES:-}" ]] && analyze_args+=(--target-inferences "$TARGET_INFERENCES")
analyze "$SWEEP_DIR" ${analyze_args[@]+"${analyze_args[@]}"}

# ---- 2. Long-clip side run (Whisper, sequential long-form) -------------------
side_ok=-   # - = not run, 1 = ok, 0 = failed
if [[ $main_ok == 1 && "$LONGFORM_RUN" == "1" ]]; then
  # Union of long ids over every main run (one results pickle per run)
  long_ids=$(find_in_main_results find_long_ids.py --min-duration "$LONGFORM_MIN_S")
  n_long=$(tr ',' '\n' <<< "$long_ids" | sed '/^$/d' | wc -l | tr -d ' ')

  # Whisper models only: wav2vec2 is not chunked, so it has nothing to compare
  models="${LONGFORM_MODELS:-$(whisper_models)}"

  if [[ $n_long -eq 0 ]]; then
    log "No samples with a clip > ${LONGFORM_MIN_S}s: skipping the side run"
  elif [[ -z "$models" ]]; then
    log "No Whisper models found for the side run: skipping it"
  else
    log "Side run $SIDE_ID: $n_long samples, models $models"
    # The -e flags in EXTRA_DOCKER_ARGS come after PER_RUN_ENV, so they win.
    # SAMPLE_IDS and HF_MODELS contain commas, which PER_RUN_ENV would split on.
    SWEEP_ID="$SIDE_ID" CONFIGS="WHISPER_LONGFORM=sequential" REPEATS=1 N_INFERENCES= \
    EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-} -e SAMPLE_IDS=$long_ids -e NUMBER_OF_SAMPLES=$n_long -e HF_MODELS=$models -e RUN_POSTFIX=${SIDE_ID} -e CHECKPOINT_SIZE=5" \
      "$SCRIPT_DIR/run_sweep.sh" || log "run_sweep.sh (side) exited with $?"
    if sweep_ok "$SIDE_DIR"; then side_ok=1; else side_ok=0; notify "side run failed" "Long-clip side run $SIDE_ID failed. Main results are fine."; fi
    analyze "$SIDE_DIR"
  fi
fi

# ---- 2b. Looped-clip side run (Whisper, temperature fallback) ----------------
# Same clips and decoding as the main run, plus OpenAI-style temperature fallback, so each
# looped transcript can be compared with what the model gives when it is pushed out of the loop.
loop_ok=-   # - = not run, 1 = ok, 0 = failed
if [[ $main_ok == 1 && "$LOOP_RUN" == "1" ]]; then
  clips=$(find_in_main_results find_loop_clips.py --min-ratio "$LOOP_MIN_RATIO")
  n_clips=$(tr ',' '\n' <<< "$clips" | sed '/^$/d' | wc -l | tr -d ' ')
  n_samples=$(tr ',' '\n' <<< "$clips" | sed '/^$/d' | cut -d'|' -f1 | sort -u | wc -l | tr -d ' ')
  models="${LOOP_MODELS:-$(whisper_models)}"

  if [[ $n_clips -eq 0 ]]; then
    log "No looped transcripts: skipping the loop side run"
  elif [[ -z "$models" ]]; then
    log "No Whisper models found for the loop side run: skipping it"
  else
    log "Loop side run $LOOP_ID: $n_clips clips in $n_samples samples, models $models"
    # Most samples have one looped clip, so batch clips across samples
    SWEEP_ID="$LOOP_ID" CONFIGS="WHISPER_FALLBACK=True" REPEATS=1 N_INFERENCES= \
    EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-} -e CLIPS=$clips -e NUMBER_OF_SAMPLES=$n_samples -e HF_MODELS=$models -e RUN_POSTFIX=${LOOP_ID} -e SAMPLES_PER_CALL=20" \
      "$SCRIPT_DIR/run_sweep.sh" || log "run_sweep.sh (loops) exited with $?"
    if sweep_ok "$LOOP_DIR"; then loop_ok=1; else loop_ok=0; notify "loop side run failed" "Loop side run $LOOP_ID failed. Main results are fine."; fi
    analyze "$LOOP_DIR"
  fi
fi

# ---- 3. Upload + verify ------------------------------------------------------
# Snapshot the pipeline log into the sweep folder so it is uploaded too
cp "$PIPELINE_LOG" "$SWEEP_DIR/pipeline.log"
if ! upload "$SWEEP_DIR" "$SWEEP_ID"; then
  log "UPLOAD NOT VERIFIED. Instance left running so nothing is lost."
  notify "UPLOAD FAILED" "Upload of $SWEEP_ID not verified. Instance left running."
  NOTIFIED_EXIT=1; exit 1
fi
if [[ $main_ok == 1 ]] && ! copy_results "$SWEEP_DIR" "$SWEEP_ID"; then
  log "RESULTS COPY NOT VERIFIED. Instance left running so nothing is lost."
  notify "RESULTS COPY FAILED" "Results of $SWEEP_ID not copied next to the logs. Instance left running."
  NOTIFIED_EXIT=1; exit 1
fi
if [[ -d "$SIDE_DIR" ]]; then
  { upload "$SIDE_DIR" "$SIDE_ID" && { [[ $side_ok != 1 ]] || copy_results "$SIDE_DIR" "$SIDE_ID"; }; } \
    || { log "Side run upload or results copy not verified (main results are safe)"; side_ok=0; }
fi
if [[ -d "$LOOP_DIR" ]]; then
  { upload "$LOOP_DIR" "$LOOP_ID" && { [[ $loop_ok != 1 ]] || copy_results "$LOOP_DIR" "$LOOP_ID"; }; } \
    || { log "Loop side run upload or results copy not verified (main results are safe)"; loop_ok=0; }
fi

# ---- 4. Terminate ------------------------------------------------------------
summary="main=$([[ $main_ok == 1 ]] && echo ok || echo FAILED) side=$side_ok loops=$loop_ok"
[[ "$TERMINATE" == "1" ]] || { log "Done (TERMINATE=0). $summary"; notify "done" "$summary. TERMINATE=0, instance still running."; exit 0; }
if [[ $main_ok != 1 && "$TERMINATE_ON_FAILURE" != "1" ]]; then
  log "Main sweep failed: instance left running. Inspect, then terminate it by hand."
  NOTIFIED_EXIT=1; exit 1
fi
log "Terminating $LAMBDA_INSTANCE_ID in ${TERMINATE_DELAY}s. Cancel: kill $$ (or Ctrl-C if DETACH=0). $summary"
notify "done, terminating" "$summary. Terminating $LAMBDA_INSTANCE_ID in ${TERMINATE_DELAY}s."
sleep "$TERMINATE_DELAY"
curl -fsS -u "$API_KEY:" -H "Content-Type: application/json" \
  -d "{\"instance_ids\": [\"$LAMBDA_INSTANCE_ID\"]}" \
  "$LAMBDA_API/instance-operations/terminate"
log "Terminate request sent."
