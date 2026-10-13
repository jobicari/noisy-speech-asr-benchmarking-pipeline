#!/usr/bin/env bash
# Run the ASR container once per config (x REPEATS) on the Lambda instance and
# record, per run: wall time, nvidia-smi samples, docker CPU/mem samples, logs.
# Analyse afterwards with:  python3 analyze.py results/lambda_bench_runs/<sweep>/
#
# Usage (defaults shown; override any of them as env vars):
#   CONFIGS="BATCH_SIZE=1 BATCH_SIZE=8 BATCH_SIZE=16 BATCH_SIZE=22" \
#   REPEATS=2 N_INFERENCES=2200 ./run_sweep.sh
#
# A config is one or more VAR=value pairs joined by commas, e.g.
#   "BATCH_SIZE=16,CHECKPOINT_SIZE=25"
# Each pair is passed to the container with -e, overriding .env.docker.
set -euo pipefail

IMAGE="${IMAGE:-asr-benchmark:lambda}"
ENV_FILE="${ENV_FILE:-$HOME/noisy-speech-asr-benchmarking-pipeline/.env.docker}"
KEY_FILE="${KEY_FILE:-$HOME/gcs-key.json}"
CONFIGS="${CONFIGS:-BATCH_SIZE=1 BATCH_SIZE=8 BATCH_SIZE=16 BATCH_SIZE=22}"
REPEATS="${REPEATS:-1}"
# Total transcriptions one run performs (samples x 22 corruptions x models).
# Used for per-inference throughput when the app does not emit [TIMING] lines.
N_INFERENCES="${N_INFERENCES:-}"
# Extra env applied to every run. {run_id} is replaced per run. Use it to give
# each run its own GCS output prefix so checkpoint-resume never skips work,
# e.g. PER_RUN_ENV="RUN_NAME=bench_{run_id}" (use your app's real variable).
PER_RUN_ENV="${PER_RUN_ENV:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
GPU_SAMPLE_MS="${GPU_SAMPLE_MS:-500}"
OUT_ROOT="${OUT_ROOT:-./results/lambda_bench_runs}"
SWEEP_ID="${SWEEP_ID:-sweep_$(date -u +%Y%m%dT%H%M%SZ)}"

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
SWEEP_DIR="$OUT_ROOT/$SWEEP_ID"; mkdir -p "$SWEEP_DIR"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
cat > "$SWEEP_DIR/sweep.json" <<EOF
{"sweep_id": "$SWEEP_ID", "image": "$IMAGE", "gpu": "$gpu_name",
 "host_cpus": $(nproc), "configs": "$CONFIGS", "repeats": $REPEATS,
 "started_utc": "$(date -u +%FT%TZ)"}
EOF
echo "Sweep $SWEEP_ID on $gpu_name -> $SWEEP_DIR"

for cfg in $CONFIGS; do
  for rep in $(seq 1 "$REPEATS"); do
    cfg_slug=$(echo "$cfg" | tr ',=' '__')
    run_id="${cfg_slug}_r${rep}"
    run_dir="$SWEEP_DIR/$run_id"; mkdir -p "$run_dir"

    env_args=()
    IFS=',' read -ra pairs <<< "$cfg"
    for p in "${pairs[@]}"; do env_args+=(-e "$p"); done
    if [[ -n "$PER_RUN_ENV" ]]; then
      IFS=',' read -ra pairs <<< "${PER_RUN_ENV//\{run_id\}/$run_id}"
      for p in "${pairs[@]}"; do env_args+=(-e "$p"); done
    fi

    echo "=== $run_id  (${env_args[*]})"
    # GPU sampler: one CSV row per GPU per interval.
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits -lms "$GPU_SAMPLE_MS" > "$run_dir/gpu.csv" &
    gpu_pid=$!

    cname="asrbench_${run_id}_$$"
    t0=$(date +%s.%N)
    set +e
    $DOCKER run --name "$cname" --gpus all --env-file "$ENV_FILE" \
      -v "$KEY_FILE:/secrets/key.json:ro" "${env_args[@]}" $EXTRA_DOCKER_ARGS \
      "$IMAGE" > "$run_dir/container.log" 2>&1 &
    run_pid=$!
    # Container CPU/mem sampler (docker stats is ~1-2 s per sample).
    ( sleep 2
      while kill -0 "$run_pid" 2>/dev/null; do
        $DOCKER stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' "$cname" 2>/dev/null \
          | sed "s/^/$(date +%s.%N),/" >> "$run_dir/cpu.csv"
      done ) &
    cpu_pid=$!
    wait "$run_pid"; rc=$?
    t1=$(date +%s.%N)
    set -e
    kill "$gpu_pid" "$cpu_pid" 2>/dev/null || true; wait 2>/dev/null || true
    $DOCKER rm "$cname" >/dev/null 2>&1 || true

    # Pull structured timing lines (see TIMING_SPEC.md) out of the log.
    grep -o '\[TIMING\] .*' "$run_dir/container.log" | sed 's/^\[TIMING\] //' > "$run_dir/timing.jsonl" || true

    cat > "$run_dir/run.json" <<EOF
{"run_id": "$run_id", "config": "$cfg", "repeat": $rep, "exit_code": $rc,
 "t_start": $t0, "t_end": $t1, "n_inferences": ${N_INFERENCES:-null},
 "gpu_sample_ms": $GPU_SAMPLE_MS}
EOF
    printf '    exit=%s  wall=%.1fs  timing_lines=%s\n' "$rc" "$(awk "BEGIN{print $t1 - $t0}")" "$(wc -l < "$run_dir/timing.jsonl")"
  done
done
echo "Done. Analyse with: python3 analyze.py $SWEEP_DIR --price-per-hour <USD>"
