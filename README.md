# Noisy-Speech ASR Benchmarking Pipeline

Benchmarks how robust speech-recognition models (the Whisper family, plus wav2vec2 as a contrast) are to degraded audio. For each speech sample, the pipeline creates a set of corrupted versions (noise, reverb, dropouts, codecs, packet loss, band limiting, clipping), transcribes every version with one or more models, and records WER/CER plus audio-health metrics for each one.

It runs in two places:

- **Apple Silicon Mac (M1 Pro):** MLX backend (`mlx-whisper`), for development and smoke tests.
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

### Next
1. Model comparison with 7 HF models (see below). Smoke run on Lambda to get per-model timings, then the full run.
2. Profile one inference call to see where the 38 s goes.
3. If inference time matters with 7 models: run two models at once in separate processes, since the GPU is two-thirds idle.
4. Later: GitHub Container Registry instead of building on each instance.

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

## Running on the Mac (MLX)

One-time setup:
```bash
conda env create -f environment.yml
conda activate asr-benchmark-pipe
pip install -e .            # or prefix commands with PYTHONPATH=src
```

`.env` needs at least `INFERENCE_BACKEND=mlx`, `SPEECH_DATASET`, `IR_DATASET` and `USE_CLOUD_STORAGE=False`.

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
- On the M1 Pro (8 performance cores, 32 GB), 1 worker keeps up with inference. More workers compete with MLX for CPU and slow prep down.

---

## Running on Lambda (A10, Docker, GCS)

### 1. Instance
- Instance type: **A10 (24 GB)**, image: **Lambda Stack 22.04** (Docker, NVIDIA driver and container toolkit preinstalled). No persistent filesystem is needed, because everything goes to GCS.
- SSH: `ssh -i ~/.ssh/lambda_key ubuntu@<ip>`
- Docker without sudo, once per instance: `sudo usermod -aG docker $USER`, then reconnect.

### 2. Code and credentials
```bash
# Clone with a short-lived, repo-scoped fine-grained PAT (enter it as the password)
git clone -b exploration https://github.com/<your-github-user>/noisy-speech-asr-benchmarking-pipeline.git
cd noisy-speech-asr-benchmarking-pipeline
```
From the Mac, copy the secrets (they are gitignored):
```bash
scp -i ~/.ssh/lambda_key .env.docker ubuntu@<ip>:~/noisy-speech-asr-benchmarking-pipeline/
scp -i ~/.ssh/lambda_key /path/to/your/gcs-key.json ubuntu@<ip>:~/gcs-key.json
ssh -i ~/.ssh/lambda_key ubuntu@<ip> chmod 644 ~/gcs-key.json   # container user has a different UID
```

Re-copy `.env.docker` whenever you change it on the Mac, for example after editing `HF_MODELS`. Pulling the repo doesn't update it.

Create a host folder for the Hugging Face model cache, once per instance:
```bash
mkdir -p ~/hf-cache && chmod 777 ~/hf-cache   # container user 'app' must be able to write
```
The Dockerfile sets `HF_HOME=/app/.cache/huggingface`, but every `docker run` starts a fresh container, so without a mount each run downloads every model again (several GB with medium and large-v3) and that time lands in "fixed" seconds. Mounting `~/hf-cache` there keeps the downloads between runs.

### 3. Build
```bash
docker build -t asr-benchmark:lambda .
```
The image is `python:3.12-slim` with CUDA torch from the `cu126` wheel index, ffmpeg and libsndfile, and runs as a non-root `app` user. `src/` and `app.py` are copied in, so rebuild after every pull. Keep `asr-benchmark:local` (ARM, Mac) and `asr-benchmark:lambda` (x86) separate.

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
`run_sweep.sh` runs the container once per config and repeat. For each run it records wall time, `nvidia-smi` samples every 500 ms, `docker stats` CPU, the container log, and the extracted `[TIMING]` lines.

Example: smoke run of the current setup (one config, one repeat):
```bash
tmux new -s sweep
CONFIGS="SAMPLES_PER_CALL=1" \
ENV_FILE=$PWD/.env.docker KEY_FILE=$PWD/gcs-key.json SWEEP_ID=models7_smoke \
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

What the report shows per config: steady-state inferences/s, audio seconds per second and RTF, the GPU share of per-sample time, per-sample and per-inference latency percentiles, GPU utilisation, VRAM peak, power, container CPU, and (with a price) cost per audio hour and a projection for the target run. The run table splits wall time into fixed (model load and finalize), prep (worker CPU time), wait, fetch, inference, score and overhead.

How to read a worker sweep: `wait_s` should drop as workers increase until it reaches about 0. The smallest worker count that gets there is the one to use. Ignore the "× M1" column: its baseline mixes two model sizes.

---

## Outputs

Local: `results/transcribe_cache/`. GCS: the same relative keys in the bucket.

- `transcribe_cache/{run_label}.pkl`: the final results, a list of row dicts.
- `transcribe_cache/checkpoint/{run_label}_CP_{nnnn}.pkl`: per-checkpoint results. They are deleted once the final file is written (`clear_cache=True`).

Row fields: `idx` (sample id), `model`, `corr_type`, `wer`, `w_hits/subs/dels/ins`, `num_words`, `cer`, `c_hits/subs/dels/ins`, `num_chars`, `sample_analysis` (metrics + ASR flags), `ref`, `hyp`. `ref` and `hyp` are stored after normalization.

### Timing records
Every line starting with `[TIMING] ` is a flushed JSON object with an ISO `ts`:

| `event` | When | Fields |
|---|---|---|
| `phase` | model load (per model), finalize | `name`, `model` or `run_label`, `seconds` |
| `prep` | once per sample, printed by the worker | `sample_idx`, `n_items`, `t_prep_s` |
| `sample` | once per sample per model | `model`, `sample_idx`, `n_items`, `audio_s`, `t_infer_s`, `t_score_s`, `t_wait_s`, `group_size`. With `SAMPLES_PER_CALL` > 1, `t_infer_s` is the call time split by clip count, and `t_wait_s` is charged to the first sample of each group (0 for the others). `t_wait_s` is the same on every model's record |
| `checkpoint` | each checkpoint save | `checkpoint_idx`, `seconds` (save time), `t_fetch_s` (time to pull that checkpoint's rows from the stream) |

With MLX, `model_load` is close to 0 because models load on the first transcribe call, so the load time lands in the first sample's `t_infer_s`.

---

## Known gotchas

- **Hang after "RUN COMPLETED SUCCESSFULLY":** caused by closing a partly consumed HF streaming iterator. Don't call `iterator.close()`. `app.py` ends with `os._exit(0)` on purpose. The `resource_tracker ... leaked semaphore` warning at exit is expected and harmless.
- **Key file unreadable in the container:** the container user's UID differs from the host's, so use `chmod 644` on the host copy.
- **`docker: permission denied`** on a fresh instance: add the user to the `docker` group and reconnect.
- **Architecture:** Mac builds are ARM64 and Lambda is x86_64. Build on the instance, or use `--platform linux/amd64`.
- **Run label doesn't record code settings.** It hashes the model list but not `SAMPLES_PER_CALL`, the thread caps or the normalizer, so two runs that differ only in those share a label and the second loads the first one's results. Give each a different `RUN_POSTFIX`.
- **The normalizer needs the Hub once:** `cleanup_pipeline` downloads `normalizer.json` from `openai/whisper-small` on first use in each process (cached afterwards).
- **`docker builder prune`** wipes all build cache. Use `docker rmi <tag>` instead.
- **Torch version:** must be ≥ 2.6 from the `cu126` index (transformers needs `torch.accelerator`).
- **Spawn workers** re-import the main module. Heavy imports in `app.py` stay inside `if __name__ == "__main__":` so workers start light.
- **Corruption lambdas** can't be pickled, so each worker builds its own corruption list in `init_worker`. The IR is loaded once in the main process and passed in through `initargs`.
- **torchcodec's lazy `AudioDecoder`** can't be pickled, so samples are decoded in the main process (`_sample_decoder`) before going to the pool.

## Housekeeping
- Revoke or rotate the GitHub PAT used for cloning when you're done with an instance.
- `notebooks/tests-asr-pipeline.ipynb` still uses the old API.
- A transformers deprecation warning about passing `generation_config` alongside other kwargs is known and harmless.
