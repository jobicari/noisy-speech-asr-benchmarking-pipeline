# Noisy-Speech ASR Benchmarking Pipeline

Benchmarks how robust speech-recognition models (the Whisper family, plus wav2vec2 as a contrast) are to degraded audio. For each speech sample, the pipeline creates a set of corrupted versions (noise, reverb, dropouts, codecs, packet loss, band limiting, clipping), transcribes every version with one or more models, and records WER/CER plus audio-health metrics for each one.

It runs in two places:

- **Apple Silicon workstation (development reference: M1 Pro, 32 GB):** MLX backend (`mlx-whisper`), for development and smoke tests.
- **Lambda Labs A10 GPU:** Hugging Face Transformers backend on CUDA, inside Docker, for long runs. Results are stored in Google Cloud Storage.

---

## Current state (October 2026)

### Done
- **Batched GPU inference.** Every corruption of a sample goes to the model in one call (`BATCH_SIZE=22` = all 22 corruptions of the full list).
- **Multiple models, corrupt once.** Each sample is corrupted once and the same audio is shared by every model in `HF_MODELS` / `MLX_MODELS`. One results file per run, with a `model` column.
- **Explicit backend.** `INFERENCE_BACKEND=hf|mlx`. `REQUIRE_CUDA=True` makes the HF path fail fast when no GPU is visible.
- **CPU worker pool for prep.** Corruption and audio-health analysis run in a `spawn` multiprocessing pool (`N_PREP_WORKERS`). The main process only does inference and scoring.
- **Prefetch across checkpoints.** The pool always works on the next checkpoint while the models run on the current one. At most two checkpoints of prepped audio are in memory.
- **Reproducible corruptions.** numpy is re-seeded per sample and per corruption from `SEED`, so a given sample always gets the same noise, whichever worker processes it.
- **Timing instrumentation.** Structured `[TIMING]` JSON lines (see [Timing records](#timing-records)), read by `run_sweep.sh` and `analyze.py`.
- **Checkpoint and resume.** Results are saved every `CHECKPOINT_SIZE` samples. An interrupted run resumes from the last complete checkpoint.
- **Thread caps.** `OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=2` stop BLAS and numba thread pools from oversubscribing the CPU. This cut inference by 25% on the A10.
- **Cross-sample batching (optional).** `SAMPLES_PER_CALL=K` sends the clips of K samples to each model in one call. It gave no speed-up on the A10, so keep the default of 1.
- **Non-Whisper models.** `load_hf_transcriber` detects the model type. Whisper gets its generation config (English, transcribe). CTC models such as wav2vec2 get a plain ASR pipeline.
- **Unattended runs.** `run_and_terminate.sh` runs a sweep, analyses it, uploads it to GCS, verifies the upload and terminates the instance. `progress.sh` shows live progress and ETA (see [Unattended runs](#6-unattended-runs-progress-upload-terminate)).
- **HF model cache on the host.** `~/hf-cache` is mounted into the container, so models are downloaded once per instance instead of once per run.
- **Clips over 30 s.** Whisper only takes 30 s per call. `WHISPER_LONGFORM=chunked` (default) splits longer clips into overlapping 30 s chunks, decodes them in parallel and stitches the text back together; clips of 30 s or less are unchanged. `WHISPER_LONGFORM=sequential` uses Whisper's own long-form decoding with timestamps instead (more accurate on long audio, but it also changes decoding of short clips). Each result row stores `duration_s`. LibriSpeech test-clean has 7 clips over 30 s and 4 more just under, which reverb can push over.
- **Long-clip side run.** After the main sweep, `run_and_terminate.sh` finds every sample with a clip over 30 s (`find_long_ids.py`) and reruns them on the Whisper models with `WHISPER_LONGFORM=sequential`, so chunked and sequential decoding can be compared on the same audio.
- **Looped-clip side run.** Small Whisper models sometimes repeat a phrase until the 448-token limit on badly damaged audio, which drives WER far above 100%. The main run keeps plain greedy decoding. After it, `run_and_terminate.sh` finds every clip where a Whisper model looped (`find_loop_clips.py`: the transcript compresses more than 2.4× with zlib, the test OpenAI's Whisper uses) and reruns only those clips on the Whisper models with `WHISPER_FALLBACK=True` (OpenAI-style temperature fallback), so each looped transcript can be compared with what the model gives when pushed out of the loop.
- **Crash guard and notifications.** `run_and_terminate.sh` only terminates the instance if every run exited 0 and logged `RUN COMPLETED SUCCESSFULLY`. With `NTFY_TOPIC` set, it pushes start, failure and done alerts to the ntfy app.
- **Fair text normalization.** References and transcripts both go through Whisper's `EnglishTextNormalizer` (lowercase, no punctuation, numbers as digits, standard spellings and abbreviations), so a model that writes "20" or "Mr." is not penalised against references that spell them out. WER from runs before this change is not comparable with later runs.

### Measured so far
**A10, batch-size sweep** (`bs_sweep1`, fullDr, 20 samples, 2 models, 2 repeats, before the worker pool):

| Batch size | Wall s | Inference s | Prep s |
|---|---|---|---|
| 1 | 274 | 174 | 65 |
| 8 | 152 | 52 | 66 |
| 16 | 147 | 47 | 66 |
| 22 | 137 | 39 | 65 |

Decision: `BATCH_SIZE=22`. After that, CPU prep was the bottleneck, which is why the worker pool was built.

**M1 Pro, worker pool + prefetch** (fullDr, 10 samples, 1 worker, `CHECKPOINT_SIZE=2`): 423 s wall, of which 412 s is inference. After the first sample the main process never waits for prep (`t_wait_s` = 0). Startup is about 11 s: 7.7 s for the first dataset fetch plus 3.4 s for the first prep. On the M1 the run is limited by inference, so 1 worker is enough.

**A10 sweeps after the worker pool** (fullDr, 20 samples, 2 models, `BATCH_SIZE=22`, 2 repeats each):

| Sweep | Config | Wall s | Inference s | Note |
|---|---|---|---|---|
| `worker_sweep_bs22` | 1 / 2 / 4 workers | 90–93 | 51–55 | All the same; `wait_s` ≈ 0 after the first sample from 1 worker up. −33% vs `bs_sweep1` |
| `thread_caps` | 4 workers, no caps | 91.6 | 51.5 | Workers' BLAS/numba threads compete with the main process (~800% CPU) |
| `thread_caps` | 4 workers, caps = 1 | 80.8 | 41.5 | |
| `thread_caps` | 4 workers, caps = 2 | 78.6 | 38.6 | **Chosen.** CPU ~170% |
| `spc_sweep1` | `SAMPLES_PER_CALL` = 1 / 2 / 4 (`BATCH_SIZE` 22 / 44 / 88) | 73–92 | 38.1 / 37.8 / 37.8 | No gain. VRAM 4.7 / 7.0 / 11.4 GB. First wait grows with K |

Decision: `N_PREP_WORKERS=4`, thread caps of 2, `BATCH_SIZE=22`, `SAMPLES_PER_CALL=1`. That gives about 15 inferences/s, a projection of 0.8 h and about $1.05 for 1000 samples × 22 corruptions × 2 models.

Where the time goes now (per 20-sample run): about 38 s inference, 15–33 s fixed (model download and load, see the HF cache note below), about 14 s unexplained overhead, and about 6 s first wait plus fetch. The GPU is only ~36% used and bigger batches don't help, so the inference time is probably the HF pipeline's per-clip CPU work and the token-by-token decode loop (inferred, not profiled).

**A10, 7 models** (fullDr, 4 workers, thread caps 2, `BATCH_SIZE=22`, HF cache mounted):

| Run | Samples | Wall s | Fixed s | Inference s | Inf/s | GPU util | Note |
|---|---|---|---|---|---|---|---|
| `models7_smoke` | 20 | 220 | 68 | 130 | 20.2 | 46% | First run fills the HF cache (download in fixed time) |
| `models7_100` | 100 | 672 | 15 | 630 | 23.4 | 74% | Cache reused. Overhead stays ~16 s, so it is per run, not per sample |

Both runs show a VRAM peak of 20.5 GB, but a stuck container from an earlier run was still on the GPU at the time. With it gone, the 7 models use about 15 GB.

Inference seconds per model (`models7_smoke`, 20 samples × 22 corruptions):

| Model | Infer s |
|---|---|
| whisper-large-v3 | 40.2 |
| whisper-large-v3-turbo | 26.4 |
| whisper-medium | 24.3 |
| whisper-tiny | 12.0 |
| whisper-small | 11.5 |
| whisper-base | 11.0 |
| wav2vec2-large-960h-lv60-self | 4.5 |

- Tiny, base and small all take ~11 s although they differ 6× in size, so that is a fixed per-call cost. wav2vec2, which has no token-by-token decoding, takes 4.5 s, which points at the Whisper decode loop (inferred, not profiled).
- Turbo is slower than medium: it keeps large-v3's full 32-layer encoder, and with short transcripts the encoder dominates.
- Projection: 1000 samples ≈ 1.8 h / $2.36, 2000 samples ≈ 3.6 h / $4.70.

### In progress
- **`fullrun2000`:** fullRun, 2000 samples, 7 models (308,000 inferences). The first attempt crashed after about 36 minutes on the first clip over 30 s (`2094-142345-0008`, 31.6 s), and the old script terminated the instance anyway. Both bugs are fixed. Its checkpoints are in the bucket, so the relaunch (`SWEEP_ID=fullrun2000_resume`, same `RUN_POSTFIX`) resumes from them. Its logs are in `gs://<your-bucket>/lambda_bench_runs/fullrun2000/`. Combine the timing of both attempts with `analyze.py --combine`.

### Next
1. Analyse the 2000-sample results: WER/CER per model × corruption, the size curve, and wav2vec2 vs Whisper.
2. Optional speed work: profile one Whisper call, or run two models at once in separate processes (the GPU still has headroom).
3. Later: build and push the image from CI instead of from a workstation.

### Model comparison
The research question is how noise robustness scales with model size, with one model trained differently as a contrast:

| Model | Params | Why |
|---|---|---|
| `openai/whisper-tiny` | 39M | Size curve |
| `openai/whisper-base` | 74M | Size curve |
| `openai/whisper-small` | 244M | Size curve |
| `openai/whisper-medium` | 769M | Size curve |
| `openai/whisper-large-v3` | 1.55B | Size curve |
| `openai/whisper-large-v3-turbo` | 809M | large-v3 encoder with a 4-layer decoder: does the speed trick cost robustness? |
| `facebook/wav2vec2-large-960h-lv60-self` | 317M | Fine-tuned on LibriSpeech (in-domain for the test set). Expected to be very good on clean audio and to drop sharply under noise |

wav2vec2 only runs on the HF backend. The medium and large models are much slower than turbo, because their decoders have 24–32 layers instead of 4.

---

## How it works

```
app.py
 ├─ load dataset (LibriSpeech test-clean, streaming) and the MIT impulse response (once)
 ├─ start a spawn Pool(N_PREP_WORKERS); each worker builds its own corruption list
 ├─ load every model in the model list
 └─ run_checkpoint_loop (models.py)
      for each checkpoint of CHECKPOINT_SIZE samples:
        fetch checkpoint k+1 from the stream, start its prep in the pool (prefetch)
        run_inference on checkpoint k:
          for each group of SAMPLES_PER_CALL prepped samples (in order):   ← main process waits here if prep lags
            for each model: transcribe all clips of the group in one call,
                            split back per sample, score WER/CER
        save checkpoint k
      concatenate checkpoints into one results file
```

| Module | Role |
|---|---|
| `app.py` | Entry point: dataset, IR, pool, models, run |
| `src/asr_pipeline_benchmark/configuration.py` | `Config` dataclass, read from environment / `.env`, validated in `__post_init__` |
| `.../prep.py` | Worker code: `init_worker(test_type, ir)`, `prep_sample(sample)` (corrupt + audio health + prep timing) |
| `.../models.py` | Model loaders (`load_transcriber_engine`), `run_inference`, `run_checkpoint_loop` (prefetch, checkpoints, resume, finalize) |
| `.../preprocessing.py` | Corruption functions and the `CorruptionListShort` / `CorruptionListFull` sets |
| `.../audio_analysis.py` | Audio-health metrics (RMS, peak, crest factor, ZCR, MFCC stats, flags) |
| `.../data.py` | Lazy loaders for the speech dataset (streaming) and the MIT IR dataset |
| `.../storage.py` | `LocalStorage` / `GCSStorage` with the same interface |
| `.../utility.py` | Run label, `checkpoint_iterator`, `print_timing`, seeding |
| `run_sweep.sh`, `analyze.py` | Benchmark tooling (repo root) |

### Test types and corruptions
| `TEST_TYPE` | Corruptions | Use |
|---|---|---|
| `smallDr` | 3: `none`, white noise 3 dB SNR, synthetic reverb | Quick smoke tests |
| `fullDr` | 22: the full list | Dress rehearsal, benchmarks |
| `fullRun` | 22: the full list | Production run |

The full list contains: clean, white noise, synthetic and real (MIT IR #251) reverb, smart dropouts, a combined noise+reverb+dropout chain, Opus codec at 8k and 6k, five packet-loss levels, three band limits, and three levels each of hard clipping and normalised clipping.

---

## Configuration

Everything is set through environment variables. Locally they come from `.env` (python-dotenv). In Docker they come from `.env.docker` via `--env-file`. Variables given on the command line or with `docker run -e` override the file (`load_dotenv(override=False)`).

> `.env.docker` must not quote values: Docker's `--env-file` keeps the quotes.

| Variable | Example | Meaning |
|---|---|---|
| `INFERENCE_BACKEND` | `hf` / `mlx` | Backend. `mlx` only on Apple Silicon |
| `REQUIRE_CUDA` | `True` | HF path raises if no CUDA device (set `False` for CPU/MPS testing) |
| `HF_MODELS` | `openai/whisper-small,openai/whisper-large-v3-turbo` | Comma-separated, no quotes. Whisper and CTC models (wav2vec2) both work. It can't be passed through `run_sweep.sh` (the script splits on commas), so set it in `.env.docker` |
| `MLX_MODELS` | `mlx-community/whisper-small-mlx,mlx-community/whisper-large-v3-turbo` | Comma-separated, no quotes |
| `TEST_TYPE` | `smallDr` / `fullDr` / `fullRun` | Required, no default |
| `NUMBER_OF_SAMPLES` | `20` | Samples taken from the stream |
| `BATCH_SIZE` | `22` | Inference batch size (HF only; MLX loops one at a time) |
| `CHECKPOINT_SIZE` | `50` | Samples per checkpoint, which is also the prefetch unit |
| `N_PREP_WORKERS` | `1` to `16` | Size of the prep pool. `4` on the A10 |
| `SAMPLES_PER_CALL` | `1` | Prepped samples whose clips go into one model call. Groups never cross a checkpoint; the last group of a checkpoint can be smaller. Set `BATCH_SIZE` to 22 × K to match |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` | `2` | Thread caps for every process (workers and main). Use `2` on the A10 |
| `WHISPER_LONGFORM` | `chunked` / `sequential` | How Whisper handles clips over 30 s (see [Done](#done)). Default `chunked` |
| `WHISPER_FALLBACK` | `False` | `True` turns on OpenAI-style temperature fallback for Whisper: temperatures 0 to 1 in steps of 0.2, `compression_ratio_threshold=1.35` (token space, the equivalent of 2.4 on text) and `logprob_threshold=-1.0`. Seeded from `SEED`. Used by the looped-clip side run |
| `CLIPS` | `6930-75918-0003\|BL_500_1000,6930-75918-0004\|none` | Only run these clips, as `sample_id\|corr_type` pairs joined by commas. Overrides `SAMPLE_IDS`; set `NUMBER_OF_SAMPLES` to the number of distinct samples. Pass it with `EXTRA_DOCKER_ARGS`, like `SAMPLE_IDS` |
| `SAMPLE_IDS` | `2094-142345-0008,1995-1836-0004` | Only run these sample ids (the stream is filtered before `NUMBER_OF_SAMPLES` is applied, so set that to the id count). Contains commas, so pass it with `EXTRA_DOCKER_ARGS="-e SAMPLE_IDS=..."`, not `PER_RUN_ENV` |
| `SEED` | `42` | Base seed for the corruptions |
| `RUN_POSTFIX` | `bs22_w4_r1` | Appended to the run label. Use a new one for every benchmark run |
| `FORCE_RETRAIN` | `False` | `True` ignores a cached results file with the same label |
| `RUN_BENCHMARK_INFERENCE` | `True` | `False` only loads cached results |
| `USE_CLOUD_STORAGE` | `True` | GCS instead of the local `results/` folder |
| `GCS_BUCKET_NAME` | `your-gcs-bucket-name` | |
| `GOOGLE_APPLICATION_CREDENTIALS` | `/secrets/key.json` | Path **inside** the container |
| `SPEECH_DATASET` | `openslr/librispeech_asr` | |
| `IR_DATASET` | `benjamin-paine/mit-impulse-response-survey` | |

**Run label:** `{TEST_TYPE}__{backend}__{8-char hash of sorted model list}__{N}samples[__{RUN_POSTFIX}]`. If a results file with the same label exists, it is loaded instead of running again, and partial checkpoints with the same label are resumed. Change `RUN_POSTFIX` (or set `FORCE_RETRAIN=True`) to force a new run.

---

## Running on Apple Silicon (MLX)

One-time setup:
```bash
conda env create -f environment.yml
conda activate asr-benchmark-pipe
pip install -e .            # or prefix commands with PYTHONPATH=src
```

Create `.env` from the tracked template and fill in the placeholders (HF token, bucket, key path): `cp .env.example .env`. For local runs it needs at least `INFERENCE_BACKEND=mlx`, `SPEECH_DATASET`, `IR_DATASET` and `USE_CLOUD_STORAGE=False`.

Smoke test (about 1 minute):
```bash
TEST_TYPE=smallDr NUMBER_OF_SAMPLES=5 N_PREP_WORKERS=1 CHECKPOINT_SIZE=2 \
RUN_POSTFIX=smoke_$(date +%H%M) \
python -u app.py 2>&1 | tee results/logs/smoke.log
```

fullDr check (exercises the IR path and all 22 corruptions):
```bash
TEST_TYPE=fullDr NUMBER_OF_SAMPLES=2 N_PREP_WORKERS=1 RUN_POSTFIX=fulldr_check \
python -u app.py 2>&1 | tee results/logs/fulldr_check.log
```

What to check:
- the log ends with `RUN COMPLETED SUCCESSFULLY`;
- rows = samples × corruptions × models;
- one `prep` record per sample, one `sample` record per sample per model;
- after the first sample, `t_wait_s` is close to 0;
- the same config with the same `SEED` gives identical WER on rerun.

Notes:
- Use `python -u` (or `PYTHONUNBUFFERED=1`) when piping to `tee`. Otherwise plain `print` lines are buffered and show up out of order next to the flushed `[TIMING]` lines.
- `tee file` overwrites and `tee -a file` appends. Use one log file per run.
- Always pass `N_PREP_WORKERS` explicitly. `RUN_POSTFIX` is only a label and does not set anything.
- On an M1 Pro (8 performance cores, 32 GB), 1 worker keeps up with inference. More workers compete with MLX for CPU and slow prep down.

---

## Running on Lambda (A10, Docker, GCS)

### 1. Instance
- Instance type: **A10 (24 GB)**, image: **Lambda Stack 22.04** (Docker, NVIDIA driver and container toolkit preinstalled). No persistent filesystem is needed, because everything goes to GCS.
- SSH: `ssh -i ~/.ssh/lambda_key ubuntu@<ip>`
- Docker without sudo, once per instance: `sudo usermod -aG docker $USER`, then reconnect.

### 2. Code and credentials
```bash
# Clone with a short-lived, repo-scoped fine-grained PAT (enter it as the password)
git clone -b main https://github.com/<your-github-user>/noisy-speech-asr-benchmarking-pipeline.git
cd noisy-speech-asr-benchmarking-pipeline
```
`.env.docker` is created on the workstation from the tracked template (`cp .env.docker.example .env.docker`, then fill in the HF token and bucket). Clone the repo on the instance first: `.env.docker` goes into the repo folder there, next to `run_and_terminate.sh`. Without it every run fails at once with `exit=125` and `open .../.env.docker: no such file or directory`. From the repo folder on the workstation, copy the secrets (they are gitignored):
```bash
scp -i ~/.ssh/lambda_key .env.docker ubuntu@<ip>:~/noisy-speech-asr-benchmarking-pipeline/.env.docker
ssh -i ~/.ssh/lambda_key ubuntu@<ip> chmod 600 ~/noisy-speech-asr-benchmarking-pipeline/.env.docker
scp -i ~/.ssh/lambda_key /path/to/your/gcs-key.json ubuntu@<ip>:~/gcs-key.json
ssh -i ~/.ssh/lambda_key ubuntu@<ip> chmod 644 ~/gcs-key.json   # container user has a different UID
```
If secrets are kept in a password manager with a command-line client (the examples use 1Password's `op`), pipe them over ssh instead, so they never land on the workstation's disk (`scp` can't read from a pipe). Quote the reference: item and field names may contain spaces. With a host alias from `~/.ssh/config`, drop `-i` and `ubuntu@<ip>`:
```bash
op read -n "op://<vault>/<item>/<field>" | ssh <host-alias> 'umask 077; cat > ~/noisy-speech-asr-benchmarking-pipeline/.env.docker'
op read "op://<vault>/<item>/<file>.json" | ssh <host-alias> 'umask 077; cat > ~/gcs-key.json && chmod 644 ~/gcs-key.json'
op read -n "op://<vault>/<item>/<field>" | ssh <host-alias> 'umask 077; cat > ~/lambda-api-key'
```
`umask 077` makes the file private from the moment it is created; only the GCS key is then opened to `644`. For a 1Password Document item, use `op document get "<item name>" --vault <vault>` in place of `op read`, which only takes `op://` references.

Where the scripts expect each file on the instance (these are the defaults):

| File | Path | Mode |
|---|---|---|
| GCS service-account key | `~/gcs-key.json` (`KEY_FILE`) | `644` |
| Lambda API key | `~/lambda-api-key` (`LAMBDA_API_KEY_FILE`) | `600` |
| `.env.docker` | `~/noisy-speech-asr-benchmarking-pipeline/.env.docker` (`ENV_FILE`) | |
| HF model cache | `~/hf-cache` | `777` |

Re-copy `.env.docker` whenever you change it on the workstation, for example after editing `HF_MODELS`. Pulling the repo doesn't update it.

Create a host folder for the Hugging Face model cache, once per instance:
```bash
mkdir -p ~/hf-cache && chmod 777 ~/hf-cache   # container user 'app' must be able to write
```
The Dockerfile sets `HF_HOME=/app/.cache/huggingface`, but every `docker run` starts a fresh container, so without a mount each run downloads every model again (several GB with medium and large-v3) and that time lands in "fixed" seconds. Mounting `~/hf-cache` there keeps the downloads between runs.

### 3. Image

The image is `python:3.12-slim` with CUDA torch from the `cu126` wheel index (all library versions pinned, see [Known gotchas](#known-gotchas)), ffmpeg and libsndfile, and runs as a non-root `app` user. `src/` and `app.py` are copied in, so the image must be rebuilt after every code change. Every command below uses the local tag `asr-benchmark:lambda` (the scripts' default `IMAGE`), whichever way you get the image.

**Build on a workstation and push to Docker Hub** (once per code change). On Apple Silicon, Docker builds ARM by default, so ask for x86. Build from a clean checkout of the commit the instance will clone, and tag the image with that commit:
```bash
git status --porcelain              # must print nothing
REV=$(git rev-parse HEAD)
docker login
docker buildx build --platform linux/amd64 \
  --label org.opencontainers.image.revision=$REV \
  -t <dockerhub-user>/asr-benchmark:${REV:0:7} -t <dockerhub-user>/asr-benchmark:lambda \
  --push .
```
The `revision` label is how a sweep finds out which code is in the image (`run_sweep.sh` writes it to `sweep.json` as `image_revision`). The `:lambda` tag always points at the latest push; the commit tag never moves. The image is about 4 GB. If the Docker Hub repository is private, the instance needs `docker login` too.

**On the instance: pull** the image for the commit you cloned, and give it the local tag:
```bash
REV=$(git rev-parse HEAD)
docker pull <dockerhub-user>/asr-benchmark:${REV:0:7}
docker tag <dockerhub-user>/asr-benchmark:${REV:0:7} asr-benchmark:lambda
docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' asr-benchmark:lambda   # must print $REV
```
If the pull fails with "not found", no image was pushed for that commit: push one from the workstation, or build on the instance.

**Or build on the instance** (no Docker Hub, about 10 minutes, and the same pinned versions):
```bash
docker build --label org.opencontainers.image.revision=$(git rev-parse HEAD) -t asr-benchmark:lambda .
```

Keep `asr-benchmark:local` (ARM, Apple Silicon) and `asr-benchmark:lambda` (x86) separate.

### 4. Single run
```bash
docker run --rm --gpus all --env-file .env.docker \
  -v ~/gcs-key.json:/secrets/key.json:ro \
  -v ~/hf-cache:/app/.cache/huggingface \
  -e TEST_TYPE=fullDr -e NUMBER_OF_SAMPLES=20 -e BATCH_SIZE=22 \
  -e N_PREP_WORKERS=4 -e CHECKPOINT_SIZE=5 -e RUN_POSTFIX=single_$(date +%H%M) \
  asr-benchmark:lambda 2>&1 | tee run.log
```
Run long jobs inside `tmux` so a dropped SSH connection doesn't kill them.

### 5. Benchmark sweeps on Lambda
`run_sweep.sh` runs the container once per config and repeat. For each run it records wall time, `nvidia-smi` samples every 500 ms, `docker stats` CPU, the container log, and the extracted `[TIMING]` lines. `sweep.json` records the GPU, the image id, the commit the image was built from (`image_revision`, from its revision label), and the git commit of the checkout (with a dirty flag), so a sweep can be traced back to its code. The two commits should match.

Example: smoke run of the current setup (one config, one repeat):
```bash
tmux new -s sweep
CONFIGS="SAMPLES_PER_CALL=1" \
ENV_FILE=$PWD/.env.docker KEY_FILE=$HOME/gcs-key.json SWEEP_ID=models7_smoke \
EXTRA_DOCKER_ARGS="-v $HOME/hf-cache:/app/.cache/huggingface" \
PER_RUN_ENV="RUN_POSTFIX=models7_smoke_{run_id},N_PREP_WORKERS=4,BATCH_SIZE=22,TEST_TYPE=fullDr,NUMBER_OF_SAMPLES=20,CHECKPOINT_SIZE=20,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,NUMBA_NUM_THREADS=2" \
REPEATS=1 ./run_sweep.sh
```
To compare settings, put one config per variant in `CONFIGS`, e.g. `CONFIGS="N_PREP_WORKERS=1 N_PREP_WORKERS=4"`, and use `REPEATS=2`.
- `CONFIGS`: space-separated configs. One config is one or more `VAR=value` pairs joined by commas.
- `PER_RUN_ENV`: applied to every run; `{run_id}` is replaced per run, which gives each run its own label.
- **Always put the sweep name in `RUN_POSTFIX`** (`<sweep>_{run_id}`). `{run_id}` alone repeats across sweeps (e.g. `N_PREP_WORKERS_4_r1`), so a later run with the same label loads the earlier results from GCS instead of running inference. You can spot this by a very short wall time and `Loaded test results from cache` in `container.log`.
- Pass `ENV_FILE` and `KEY_FILE` as absolute paths (`$PWD/...`). A wrong path makes `docker run` fail with exit code 125 and a wall time of 0 s; the reason is in each run's `container.log`.
- Run it from the repo root (`OUT_ROOT` is relative), inside `tmux`. To watch a run live: `tail -f results/lambda_bench_runs/<sweep>/<run_id>/container.log` or `docker logs -f $(docker ps -q | head -1)`.
- Other overrides: `IMAGE` (default `asr-benchmark:lambda`), `ENV_FILE`, `KEY_FILE` (default `~/gcs-key.json`), `OUT_ROOT` (default `./results/lambda_bench_runs`), `SWEEP_ID`, `N_INFERENCES`, `GPU_SAMPLE_MS`, `EXTRA_DOCKER_ARGS`.

Analyse:
```bash
python3 analyze.py results/lambda_bench_runs/sweep_<id> --price-per-hour <A10 $/h> --target-inferences <samples*22*models>
```
This writes `summary.csv`, `runs.csv`, `report.md` and `report.html` into the sweep folder. Copy the folder back with `scp -r`, then **terminate the instance**.

A crashed run and its resume are two sweep folders. Don't put them in one folder: `analyze.py` averages runs of the same config, but wall time and cost must be added. Use `--combine` instead:
```bash
python3 analyze.py --combine results/lambda_bench_runs/fullrun2000 results/lambda_bench_runs/fullrun2000_resume \
  --price-per-hour 1.29 --out fullrun2000_combined.csv
```
It writes one row per attempt and a TOTAL row: wall, fixed, inferences, audio seconds and cost are added, and the rates are recomputed from the sums. Failed attempts count (their GPU time was paid for). `duplicated_inferences` counts work redone after the resume.

What the report shows per config: steady-state inferences/s, audio seconds per second and RTF, the GPU share of per-sample time, per-sample and per-inference latency percentiles, GPU utilisation, VRAM peak, power, container CPU, and (with a price) cost per audio hour and a projection for the target run. The run table splits wall time into fixed (startup, model load, resume skip and finalize), prep (worker CPU time), wait, fetch, inference, score and overhead.

How to read a worker sweep: `wait_s` should drop as workers increase until it reaches about 0. The smallest worker count that gets there is the one to use. Ignore the "× M1" column: its baseline mixes two model sizes.

### 6. Unattended runs: progress, upload, terminate
`run_and_terminate.sh` wraps `run_sweep.sh` for long runs:
1. Main sweep, then a check that every run exited 0 and logged `RUN COMPLETED SUCCESSFULLY`.
2. Long-clip side run (only if the main sweep succeeded): `find_long_ids.py` reads the results pickle named in `container.log` and lists every sample with a clip over `LONGFORM_MIN_S` (default 30). Those samples are rerun on the Whisper models from `HF_MODELS` (or `LONGFORM_MODELS`) with `WHISPER_LONGFORM=sequential`, into the sweep `<SWEEP_ID>_longform`.
3. Looped-clip side run (only if the main sweep succeeded): `find_loop_clips.py` lists every (sample, corruption) clip where a Whisper model's transcript compresses more than `LOOP_MIN_RATIO` (default 2.4). Those clips are rerun on the Whisper models from `HF_MODELS` (or `LOOP_MODELS`) with `WHISPER_FALLBACK=True` and `SAMPLES_PER_CALL=20`, into the sweep `<SWEEP_ID>_loops`. fullrun2000 had 1,077 such clips in 959 samples.
4. `analyze.py` on all sweeps.
5. Upload them to `gs://<bucket>/lambda_bench_runs/` and verify every file. A snapshot of the pipeline log is uploaded as `<SWEEP_ID>/pipeline.log`. Then each successful run's results pickle is copied inside GCS to `lambda_bench_runs/<sweep>/<run_id>/` (`copy_results.py`), so results and timing live together. A failed copy of the main results keeps the instance up.
6. Terminate, only if the main sweep succeeded and its upload verified. A failed side run is reported but does not block termination.

Shutting down from inside the instance (`sudo shutdown`) does **not** stop Lambda billing; only terminating does.

Extra variables (on top of `run_sweep.sh`'s): `TERMINATE` (1/0), `TERMINATE_DELAY` (default 120 s), `TERMINATE_ON_FAILURE` (default 0; 1 terminates even after a failed main sweep, to cap cost), `LONGFORM_RUN` (default 1; 0 skips the side run), `LONGFORM_MIN_S`, `LONGFORM_MODELS`, `LOOP_RUN` (default 1; 0 skips the looped-clip run), `LOOP_MIN_RATIO`, `LOOP_MODELS`, `NTFY_TOPIC`, `STALL_MIN` (default 30; 0 turns the stall alert off), `PRICE_PER_HOUR`, `TARGET_INFERENCES`, `GCS_PREFIX`.

**Notifications:** install the ntfy app (iOS/Android), subscribe to a hard-to-guess topic name, and put `NTFY_TOPIC=<topic>` in `.env.docker` (or set it in the shell). Alerts: started, main run failed, side run failed, upload failed, results copy failed, no log output for `STALL_MIN` minutes (the run may be stuck; it is not stopped), script exited unexpectedly, done. No account is needed, but anyone who knows the topic can read it, so keep the name private.

One-time setup per instance:
1. Create an API key in the Lambda Cloud console (API keys page). Copy it to the instance:
   `scp -i ~/.ssh/lambda_key lambda-api-key ubuntu@<ip>:~/lambda-api-key && ssh -i ~/.ssh/lambda_key ubuntu@<ip> chmod 600 ~/lambda-api-key`. Revoke the key when you no longer need it.
2. Check the key works. It should print this instance's `id` and `ip` (ignore the nested UUID-style `id`s, which belong to attached items):
   ```bash
   curl -fsS -u "$(cat ~/lambda-api-key):" https://cloud.lambdalabs.com/api/v1/instances | python3 -m json.tool | grep -E '"(id|ip)"'
   ```
3. The script finds this instance's ID by matching its public IP (`curl https://checkip.amazonaws.com`) in that list. If that fails, set `LAMBDA_INSTANCE_ID` (shown in the console).

Tested end to end: the ID lookup and terminate call were run on a throwaway instance, which found its own ID (not the main instance's) and was terminated. The upload and verification were tested with `TERMINATE=0` on a 10-sample run.

Before starting, `docker ps` must list nothing: a leftover container shares the GPU and slows the run.

**It survives a dropped SSH connection.** The script re-runs itself detached from the terminal (`setsid nohup`) and returns at once, printing its pid and the log to follow (`tail -f results/lambda_bench_runs/<SWEEP_ID>.pipeline.log`). Stop it with `pkill -f run_and_terminate.sh`; a container that is already running keeps going until you `docker stop` it. `DETACH=0` keeps it in the foreground (then use tmux). Each container is started with `docker run -d` and collected with `docker wait`, so its exit code is recorded even if the script's docker client goes away. The first `fullrun2000_resume` lost its wrapper this way: it ran in a plain SSH shell, the connection dropped after 2 h, and the container finished with nothing left to upload or terminate.

Test run (no terminate, a few minutes):
```bash
SWEEP_ID=automation_test TERMINATE=0 CONFIGS="SAMPLES_PER_CALL=1" \
ENV_FILE=$PWD/.env.docker KEY_FILE=$HOME/gcs-key.json \
EXTRA_DOCKER_ARGS="-v $HOME/hf-cache:/app/.cache/huggingface" \
PER_RUN_ENV="RUN_POSTFIX=automation_test_{run_id},N_PREP_WORKERS=4,BATCH_SIZE=22,TEST_TYPE=fullDr,NUMBER_OF_SAMPLES=10,CHECKPOINT_SIZE=5,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,NUMBA_NUM_THREADS=2" \
REPEATS=1 ./run_and_terminate.sh
```
It should end with `VERIFIED N files in gs://.../lambda_bench_runs/automation_test/`.

Smoke test after a new image (built or pulled; no terminate, a few minutes). It runs one short and one long clip on all 7 models, then the side run on the long one:
```bash
SWEEP_ID=smoke_longclip TERMINATE=0 CONFIGS="SAMPLES_PER_CALL=1" \
ENV_FILE=$PWD/.env.docker KEY_FILE=$HOME/gcs-key.json \
EXTRA_DOCKER_ARGS="-v $HOME/hf-cache:/app/.cache/huggingface -e SAMPLE_IDS=2094-142345-0007,2094-142345-0008" \
PER_RUN_ENV="RUN_POSTFIX=smoke_longclip_{run_id},N_PREP_WORKERS=4,BATCH_SIZE=22,TEST_TYPE=fullRun,NUMBER_OF_SAMPLES=2,CHECKPOINT_SIZE=2,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,NUMBA_NUM_THREADS=2" \
REPEATS=1 ./run_and_terminate.sh
```
Main run: 2 × 22 × 7 = 308 inferences. Side run: 1 × 22 × 6 = 132. Check:
1. `results/lambda_bench_runs/smoke_longclip/*/container.log` ends with `RUN COMPLETED SUCCESSFULLY` and has no "more than 3000 mel input features" error (chunking works on CUDA).
2. The pipeline log shows `Main sweep succeeded`, then `Side run smoke_longclip_longform: 1 samples` with the 6 Whisper models.
3. In `smoke_longclip_longform/*/container.log`, `run_label` ends with `smoke_longclip_longform`. That proves the side run's `-e` overrides (they come after `PER_RUN_ENV`, and Docker keeps the last value) took effect.
4. The log ends with `Done (TERMINATE=0). main=ok side=1`, and both folders say `VERIFIED`.
5. The ntfy app got "started" and "done".

Smoke test of the looped-clip run (no terminate, a few minutes). In fullrun2000, whisper-tiny looped on `BL_500_1000` for both samples:
```bash
SWEEP_ID=smoke_loops TERMINATE=0 LONGFORM_RUN=0 CONFIGS="SAMPLES_PER_CALL=1" \
ENV_FILE=$PWD/.env.docker KEY_FILE=$HOME/gcs-key.json \
EXTRA_DOCKER_ARGS="-v $HOME/hf-cache:/app/.cache/huggingface -e SAMPLE_IDS=6930-75918-0003,6930-75918-0004" \
PER_RUN_ENV="RUN_POSTFIX=smoke_loops_{run_id},N_PREP_WORKERS=4,BATCH_SIZE=22,TEST_TYPE=fullRun,NUMBER_OF_SAMPLES=2,CHECKPOINT_SIZE=2,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,NUMBA_NUM_THREADS=2" \
REPEATS=1 ./run_and_terminate.sh
```
Check that the pipeline log shows `Loop side run smoke_loops_loops: N clips in M samples`, that `smoke_loops_loops/*/container.log` ends with `RUN COMPLETED SUCCESSFULLY` and its `run_config` record has `"WHISPER_FALLBACK": true`, that the looped transcripts in its results differ from the main run's, and that the log ends with `loops=1` and a `VERIFIED copy` line per run.

Optional guard test: rerun with `-e HF_MODELS=bad/model` added to `EXTRA_DOCKER_ARGS` (and a new `SWEEP_ID`). Expect a "MAIN RUN FAILED" alert, no side run, and `Main sweep failed: instance left running`.

Full run (2000 samples × 22 corruptions × 7 models = 308,000 inferences, about 4 h and $5.20 on an A10 at $1.29/h, measured on fullrun2000):
```bash
SWEEP_ID=fullrun2000b CONFIGS="SAMPLES_PER_CALL=1" TARGET_INFERENCES=308000 \
ENV_FILE=$PWD/.env.docker KEY_FILE=$HOME/gcs-key.json \
EXTRA_DOCKER_ARGS="-v $HOME/hf-cache:/app/.cache/huggingface" \
PER_RUN_ENV="RUN_POSTFIX=fullrun2000b_{run_id},N_PREP_WORKERS=4,BATCH_SIZE=22,TEST_TYPE=fullRun,NUMBER_OF_SAMPLES=2000,CHECKPOINT_SIZE=50,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,NUMBA_NUM_THREADS=2" \
REPEATS=1 ./run_and_terminate.sh
```
1. It returns at once with `Running detached (pid N)`. `tail -f results/lambda_bench_runs/fullrun2000b.pipeline.log` must show `Will terminate instance <id>`.
2. `watch -n 30 ./progress.sh` (in tmux if you like) and check that samples count up.
3. Close SSH whenever you want; the run doesn't depend on it.
4. When the ETA has passed, check the Lambda console. If the instance is still there, terminate it by hand and read `results/lambda_bench_runs/<SWEEP_ID>.pipeline.log`.

**Resuming after a crash:** keep `RUN_POSTFIX` exactly the same (it is part of the run label, which picks the checkpoints to resume from) but use a new `SWEEP_ID`, e.g. `fullrun2000_resume`. `SWEEP_ID` only names the log folder, and reusing it would overwrite the crashed attempt's logs in GCS. `container.log` should say `Resuming pipeline from checkpoint N`. The checkpoint that was running at the crash was never saved, so it is redone. Combine both attempts' timing with `analyze.py --combine`.

Manual terminate from inside an instance (what the script does):
```bash
API_KEY=$(tr -d '[:space:]' < ~/lambda-api-key)
curl -fsS -u "$API_KEY:" -H "Content-Type: application/json" \
  -d '{"instance_ids": ["<instance id>"]}' \
  https://cloud.lambdalabs.com/api/v1/instance-operations/terminate
```
- The API key and instance ID are checked **before** the sweep starts, so a setup mistake fails in seconds, not after hours.
- The instance is terminated only if the main sweep succeeded and the upload is verified (every file present with the same size). Otherwise it stays up so nothing is lost (unless `TERMINATE_ON_FAILURE=1`).
- Before terminating it waits `TERMINATE_DELAY` seconds (default 120); `kill <pid>` (the pid is in the "Terminating" log line) cancels. `TERMINATE=0` uploads without terminating.
- `upload_results.py` refuses a folder without a `run.json` (not a sweep folder) or with `.env*`, `*key.json` or `.git` in it, so a wrong or empty path can't upload the repo and its secrets.
- The wrapper's own log is `results/lambda_bench_runs/<SWEEP_ID>.pipeline.log`, next to the sweep folder. A snapshot taken just before the upload goes to GCS as `<SWEEP_ID>/pipeline.log`; the upload and terminate lines after it stay on the instance.
- The upload runs in the benchmark image (`upload_results.py` is mounted in), so nothing extra is installed on the host.

Progress, in a second tmux pane (`Ctrl-b c`):
```bash
watch -n 30 ./progress.sh                                     # newest run under results/lambda_bench_runs
./progress.sh results/lambda_bench_runs/fullrun1               # a specific sweep
```
It reads the live `container.log` and shows models, current checkpoint, samples done / total, rate, ETA, and how long ago the last sample finished. A resumed run counts the skipped samples. It uses GNU `find`, so it runs on the instance, not on macOS.

### Tests

Run before rebuilding the image. They need no GPU, network or cloud credentials, and take about 30 s:

```bash
python -m unittest discover -s tests -t . -v
```

Use the project's environment (`conda activate asr-benchmark-pipe`), from the repo root; another environment fails at once with the list of missing packages. torch and transformers are never imported: the model loader is checked against stand-ins, and `run_and_terminate.sh` and `run_sweep.sh` run end to end against fake `docker`, `nvidia-smi` and `curl` in `tests/fakebin` (nothing is uploaded, terminated or sent). Covered:

| File | What |
|---|---|
| `tests/test_config_prep.py` | `CLIPS` parsing, `WHISPER_FALLBACK`, the prep filter, and that a rerun of one clip gets exactly the audio of the full run |
| `tests/test_models.py` | `run_checkpoint_loop`: rows, every `[TIMING]` record, checkpoints, resume (`resume_skip`), cached results, `SAMPLES_PER_CALL` grouping, `CLIPS`; WER of a loop; the Whisper loader's greedy, fallback and CTC settings |
| `tests/test_scripts.py` | `find_loop_clips.py`, `find_long_ids.py`, `copy_results.py` (fake GCS), `analyze.py` (distinct samples, fixed time, failed runs, `--combine`), the `run_config` record, `progress.sh` |
| `tests/test_run_wrapper.py` | Detached start, main run, long-clip and looped-clip side runs and their env, `sweep.json` provenance, timing extraction, uploads, results copy, notifications, failure paths, the stall alert, script permissions |

What they can't cover: real decoding on a GPU (including the fallback), GCS itself, and the Lambda API. The smoke tests in [Unattended runs](#6-unattended-runs-progress-upload-terminate) do that on the instance.

---

## Outputs

Local: `results/transcribe_cache/`. GCS: the same relative keys in the bucket.

- `transcribe_cache/{run_label}.pkl`: the final results, a list of row dicts.
- `transcribe_cache/checkpoint/{run_label}_CP_{nnnn}.pkl`: per-checkpoint results. They are deleted once the final file is written (`clear_cache=True`).

Row fields: `idx` (sample id), `model`, `corr_type`, `wer`, `w_hits/subs/dels/ins`, `num_words`, `cer`, `c_hits/subs/dels/ins`, `num_chars`, `sample_analysis` (metrics + ASR flags), `ref`, `hyp`, `duration_s` (length of that corrupted clip; missing in rows written before it was added, which were all 30 s or shorter). `ref` and `hyp` are stored after normalization.

### Timing records
Every line starting with `[TIMING] ` is a flushed JSON object with an ISO `ts`:

| `event` | When | Fields |
|---|---|---|
| `phase` | startup (dataset, IR, pool), model load (per model), resume skip, finalize | `name`, `seconds`; `model` on model load, `run_label` on finalize, `checkpoint_idx` and `samples_skipped` on resume skip |
| `run_config` | once, after the model loads | `config` (every `Config` field except the bucket, plus thread caps and `SAMPLE_IDS`), `versions` (python, torch, transformers, datasets, jiwer, numpy), `gpu` |
| `prep` | once per sample, printed by the worker | `sample_idx`, `n_items`, `t_prep_s` |
| `sample` | once per sample per model | `model`, `sample_idx`, `n_items`, `audio_s`, `t_infer_s`, `t_score_s`, `t_wait_s`, `group_size`. With `SAMPLES_PER_CALL` > 1, `t_infer_s` is the call time split by clip count, and `t_wait_s` is charged to the first sample of each group (0 for the others). `t_wait_s` is the same on every model's record |
| `checkpoint` | each checkpoint save | `checkpoint_idx`, `seconds` (save time), `t_fetch_s` (time to pull that checkpoint's rows from the stream) |

With MLX, `model_load` is close to 0 because models load on the first transcribe call, so the load time lands in the first sample's `t_infer_s`.

---

## Known gotchas

- **Hang after "RUN COMPLETED SUCCESSFULLY":** caused by closing a partly consumed HF streaming iterator. Don't call `iterator.close()`. `app.py` ends with `os._exit(0)` on purpose. The `resource_tracker ... leaked semaphore` warning at exit is expected and harmless.
- **Key file unreadable in the container:** the container user's UID differs from the host's, so use `chmod 644` on the host copy.
- **`docker: permission denied`** on a fresh instance: add the user to the `docker` group and reconnect.
- **Architecture:** Apple Silicon builds are ARM64 and Lambda is x86_64. Build on the instance, or use `--platform linux/amd64`.
- **Run label doesn't record code settings.** It hashes the model list but not `SAMPLES_PER_CALL`, the thread caps or the normalizer, so two runs that differ only in those share a label and the second loads the first one's results. Give each a different `RUN_POSTFIX`.
- **The normalizer needs the Hub once:** `cleanup_pipeline` downloads `normalizer.json` from `openai/whisper-small` on first use in each process (cached afterwards).
- **Stuck containers:** a container that hangs keeps running (and holding GPU memory) after the sweep moves on, and its run never reaches finalize, so its checkpoints stay in the bucket. Run `docker ps` before every sweep and `docker stop <name>` anything left over.
- **Local checkpoint listing** used to return every `.pkl` in the checkpoint folder, so a local run could resume from or merge another run's checkpoints. It now matches the run's own prefix (`storage.py`). GCS was never affected.
- **`docker builder prune`** wipes all build cache. Use `docker rmi <tag>` instead.
- **Library versions are pinned.** torch, torchaudio and torchcodec in the `Dockerfile` (≥ 2.6 from the `cu126` index: transformers needs `torch.accelerator`), everything else in `constraints-docker.txt`, so a rebuild installs the same code. They are the versions the v1 run used (rebuilt from PyPI release dates; transformers 5.19.0): with transformers 5.21.0, whisper-medium ran out of GPU memory on a looping sample with `BATCH_SIZE=22`. To upgrade, change the versions, rebuild, run the smoke tests, then commit. Each run logs the versions it used in its `run_config` record.
- **Clips over 30 s crash Whisper in short-form mode** ("more than 3000 mel input features"). The default `WHISPER_LONGFORM=chunked` handles them. MLX (`mlx-whisper`) does long-form by itself, so it never shows this error; test it with `INFERENCE_BACKEND=hf`.
- **A failed run used to terminate the instance.** `run_sweep.sh` always exits 0 and only records the exit code in `run.json`, so the old wrapper uploaded the partial folder, verified it and terminated. The wrapper now checks each run's exit code and success line first.
- **Spawn workers** re-import the main module. Heavy imports in `app.py` stay inside `if __name__ == "__main__":` so workers start light.
- **Corruption lambdas** can't be pickled, so each worker builds its own corruption list in `init_worker`. The IR is loaded once in the main process and passed in through `initargs`.
- **torchcodec's lazy `AudioDecoder`** can't be pickled, so samples are decoded in the main process (`_sample_decoder`) before going to the pool.

## Housekeeping
- Notes with real values (bucket, paths, instance details) belong in a `*.local.md` file, which is gitignored.
- Revoke or rotate the GitHub PAT used for cloning when you're done with an instance.
- Revoke the Lambda API key when you no longer run unattended jobs. It can terminate any of your instances.
- `notebooks/tests-asr-pipeline.ipynb` still uses the old API.
- A transformers deprecation warning about passing `generation_config` alongside other kwargs is known and harmless.
