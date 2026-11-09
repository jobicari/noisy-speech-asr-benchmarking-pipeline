# Scripts

Everything here is run from the repo root, e.g. `./scripts/cloud/run_and_terminate.sh`. The benchmark itself is [`app.py`](../app.py) and [`src/`](../src/), which is all the Docker image contains. The step-by-step guide is [Getting started](../docs/getting-started.md).

## `cloud/`: run and measure the benchmark on the GPU host

| Script | What it does | How it's started |
|---|---|---|
| [`run_and_terminate.sh`](cloud/run_and_terminate.sh) | One unattended run: main sweep, long-clip and looped-clip side runs, analysis, upload to GCS, verification, then terminates the instance | By hand on the instance |
| [`run_sweep.sh`](cloud/run_sweep.sh) | Runs the container once per config and repeat, and records wall time, GPU and CPU samples, logs and timing | By `run_and_terminate.sh`, or by hand for a sweep without upload |
| [`progress.sh`](cloud/progress.sh) | Live progress and ETA of the run that is writing a `container.log` | By hand, e.g. `watch -n 30 ./scripts/cloud/progress.sh` |
| [`analyze.py`](cloud/analyze.py) | Summarises a sweep (`summary.csv`, `report.md`, `report.html`) and ranks the configs. Stdlib only | By `run_and_terminate.sh`, or by hand on any machine |

## `container/`: helpers that run inside the benchmark image

`run_and_terminate.sh` mounts these at `/app/<name>` for one `docker run` each, so changing them doesn't need an image rebuild. They need `google-cloud-storage`, which the image has.

| Script | What it does |
|---|---|
| [`upload_results.py`](container/upload_results.py) | Uploads a sweep folder to GCS and checks every file's size. Refuses folders that look like the repo or hold secrets |
| [`copy_results.py`](container/copy_results.py) | Copies each run's results pickle next to its logs in GCS and checks the copy |
| [`find_long_ids.py`](container/find_long_ids.py) | Lists the samples with a clip over N seconds, for the long-clip side run |
| [`find_loop_clips.py`](container/find_loop_clips.py) | Lists the clips where a Whisper model looped, for the looped-clip side run. Imports `find_long_ids.py`, so the two stay in the same folder |

## `demo/`: build the demo page in `docs/demo/`

Both need the v2 sweep folders downloaded (see [`analysis/v2/README.md`](../analysis/v2/README.md)).

| Script | What it does |
|---|---|
| [`export_demo_clips.py`](demo/export_demo_clips.py) | Regenerates the clean and corrupted audio of each demo clip with the run's seeds, and checks it against the stored audio-health metrics |
| [`build_demo_data.py`](demo/build_demo_data.py) | Writes `docs/demo/data.json`: the demo clips' transcripts and the WER heatmap |

The analysis scripts for the full runs live with their reports, in [`analysis/v2/`](../analysis/v2/).
