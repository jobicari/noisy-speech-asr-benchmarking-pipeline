"""The helper scripts at the repo root: find_loop_clips.py, find_long_ids.py,
copy_results.py (against a fake google-cloud-storage), analyze.py and run_config logging."""
import csv
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import tests
from tests.helpers import capture_timing
from asr_pipeline_benchmark import utility
from asr_pipeline_benchmark.configuration import Config

REPO = tests.REPO


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_py(args, env=None, cwd=REPO):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def row(idx, model, corr, hyp, duration_s=5.0):
    return {"idx": idx, "model": model, "corr_type": corr, "hyp": hyp, "duration_s": duration_s}


class TestFindLoopClips(unittest.TestCase):
    def test_lists_unique_whisper_loop_clips(self):
        loop = "i am going to say that " * 40
        rows = [
            row("a", "openai_whisper-tiny", "BL_500_1000", loop),
            row("a", "openai_whisper-base", "BL_500_1000", loop),            # same clip, other model
            row("a", "openai_whisper-tiny", "none", "the cat sat on the mat"),
            row("b", "openai_whisper-small", "WN_3snr", "0" * 300),            # digit loop
            row("c", "facebook_wav2vec2-large", "PL_5_.5", loop),            # not Whisper
            row("d", "openai_whisper-tiny", "none", ""),                      # empty hypothesis
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.pkl"
            path.write_bytes(pickle.dumps(rows))
            out = run_py(["find_loop_clips.py", str(path)])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "a|BL_500_1000,b|WN_3snr")
        self.assertIn("3 looped Whisper transcripts, 2 clips in 2 samples", out.stderr)

    def test_threshold(self):
        sys.path.insert(0, str(REPO))
        from find_loop_clips import compression_ratio
        self.assertLess(compression_ratio("he hoped there would be stew for dinner turnips and carrots"), 2.4)
        self.assertGreater(compression_ratio("i am a good man " * 5), 2.4)
        self.assertEqual(compression_ratio(""), 0.0)


class TestFindLongIds(unittest.TestCase):
    def test_lists_samples_with_a_long_clip(self):
        rows = [row("a", "m", "none", "", 31.0), row("a", "m", "SV_0.5", "", 32.0),
                row("b", "m", "none", "", 12.0), {"idx": "c", "model": "m", "corr_type": "none"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.pkl"
            path.write_bytes(pickle.dumps(rows))
            out = run_py(["find_long_ids.py", str(path), "--min-duration", "30"])
        self.assertEqual(out.stdout.strip(), "a")
        self.assertIn("1 without duration_s", out.stderr)


FAKE_GCS = '''
import json, os
STATE = os.environ["FAKE_GCS_STATE"]

def _load():
    with open(STATE) as f:
        return json.load(f)

def _save(s):
    with open(STATE, "w") as f:
        json.dump(s, f)

class Blob:
    def __init__(self, name, size):
        self.name, self.size = name, size

class Bucket:
    def get_blob(self, key):
        s = _load()["blobs"]
        return Blob(key, s[key]) if key in s else None

    def copy_blob(self, blob, bucket, new_name):
        s = _load()
        s["blobs"][new_name] = blob.size + s.get("corrupt_copy_by", 0)
        s.setdefault("copies", []).append([blob.name, new_name])
        _save(s)

class Client:
    def bucket(self, name):
        assert name == "test-bucket"
        return Bucket()
'''


class TestCopyResults(unittest.TestCase):
    def run_copy(self, state, key, dest):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "google/cloud"
            pkg.mkdir(parents=True)
            (Path(d) / "google/__init__.py").write_text("")
            (pkg / "__init__.py").write_text("")
            (pkg / "storage.py").write_text(FAKE_GCS)
            state_path = Path(d) / "state.json"
            state_path.write_text(json.dumps(state))
            out = run_py(["copy_results.py", key, dest],
                         env={"PYTHONPATH": d, "GCS_BUCKET_NAME": "test-bucket",
                              "FAKE_GCS_STATE": str(state_path)})
            return out, json.loads(state_path.read_text())

    def test_copies_next_to_the_logs_and_verifies(self):
        key = "results/transcribe_cache/fullRun__hf__x__2000samples__v2.pkl"
        out, state = self.run_copy({"blobs": {key: 1000}}, key, "lambda_bench_runs/fullrun2000b/SPC_1_r1/")
        self.assertEqual(out.returncode, 0, out.stderr)
        dest = "lambda_bench_runs/fullrun2000b/SPC_1_r1/fullRun__hf__x__2000samples__v2.pkl"
        self.assertEqual(state["copies"], [[key, dest]])
        self.assertIn("VERIFIED copy", out.stdout)

    def test_missing_source_fails(self):
        out, state = self.run_copy({"blobs": {}}, "results/x.pkl", "dest")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("Not found", out.stderr)

    def test_size_mismatch_fails(self):
        out, _ = self.run_copy({"blobs": {"k.pkl": 10}, "corrupt_copy_by": 1}, "k.pkl", "dest")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("VERIFY FAILED", out.stderr)


def load_analyze():
    spec = importlib.util.spec_from_file_location("analyze", REPO / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_run(d, run_id, exit_code=0, t=(1000.0, 1100.0), samples=("s0", "s1", "s2"),
              models=("tiny", "base"), extra=()):
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "config": "SAMPLES_PER_CALL=1",
                                            "repeat": 1, "exit_code": exit_code, "t_start": t[0],
                                            "t_end": t[1], "n_inferences": None}))
    recs = [{"event": "phase", "name": "startup", "seconds": 4.0},
            *({"event": "phase", "name": "model_load", "model": m, "seconds": 2.0} for m in models),
            {"event": "run_config", "config": {"BATCH_SIZE": 22}, "versions": {}, "gpu": "A10"},
            *extra]
    for i, s in enumerate(samples):
        recs.append({"event": "prep", "sample_idx": s, "n_items": 22, "t_prep_s": 1.0})
        for m in models:
            recs.append({"event": "sample", "model": m, "sample_idx": s, "n_items": 22, "audio_s": 100.0,
                         "t_infer_s": 5.0, "t_score_s": 0.1, "t_wait_s": 3.0 if i == 0 else 0.0,
                         "group_size": 1})
    recs += [{"event": "checkpoint", "checkpoint_idx": 0, "seconds": 1.0, "t_fetch_s": 0.5},
             {"event": "phase", "name": "finalize", "run_label": "x", "seconds": 1.0}]
    (d / "timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "gpu.csv").write_text("2026/10/29 01:00:00.000, 0, 90, 50, 20000, 23000, 140.0\n")


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.analyze = load_analyze()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_counts_distinct_samples_and_fixed_time(self):
        write_run(self.tmp / "r1", "r1",
                  extra=[{"event": "phase", "name": "resume_skip", "checkpoint_idx": 1,
                          "samples_skipped": 50, "seconds": 6.0}])
        r = self.analyze.analyse_run(self.tmp / "r1", price=1.29)
        self.assertEqual(r["n_samples"], 3)                  # not 3 x 2 models
        self.assertEqual(r["n_inferences"], 3 * 2 * 22)
        self.assertAlmostEqual(r["fixed_s"], 4 + 2 * 2 + 6 + 1)  # startup, loads, resume skip, finalize
        self.assertAlmostEqual(r["infer_s"], 30.0)
        self.assertAlmostEqual(r["wait_s"], 3.0)             # charged once per sample
        self.assertAlmostEqual(r["fetch_s"], 0.5)
        self.assertAlmostEqual(r["run_cost_usd"], 1.29 * 100 / 3600)
        self.assertEqual(r["gpu_mem_peak_mib"], 20000)

    def test_report_files_and_failed_run_excluded_from_summary(self):
        sweep = self.tmp / "sweep"
        write_run(sweep / "ok", "ok")
        write_run(sweep / "bad", "bad", exit_code=1)
        out = run_py(["analyze.py", str(sweep), "--price-per-hour", "1.29",
                      "--target-inferences", "308000"])
        self.assertEqual(out.returncode, 0, out.stderr)
        for f in ("summary.csv", "runs.csv", "report.md", "report.html"):
            self.assertTrue((sweep / f).exists(), f)
        summary = read_csv(sweep / "summary.csv")
        self.assertEqual((summary[0]["runs"], summary[0]["failed"]), ("2", "1"))
        runs = {r["run_id"]: r for r in read_csv(sweep / "runs.csv")}
        self.assertEqual(runs["bad"]["n_samples"], "3")

    def test_combine_adds_a_crash_and_its_resume(self):
        write_run(self.tmp / "a/r1", "r1", exit_code=1, t=(0.0, 100.0), samples=("s0", "s1", "s2"))
        write_run(self.tmp / "b/r1", "r1", t=(200.0, 500.0), samples=("s2", "s3"))
        out_csv = self.tmp / "combined.csv"
        out = run_py(["analyze.py", "--combine", str(self.tmp / "a"), str(self.tmp / "b"),
                      "--price-per-hour", "1.29", "--out", str(out_csv)])
        self.assertEqual(out.returncode, 0, out.stderr)
        total = read_csv(out_csv)[-1]
        self.assertEqual(total["attempt"], "TOTAL")
        self.assertAlmostEqual(float(total["wall_s"]), 400.0)
        self.assertEqual(int(float(total["n_inferences"])), (3 + 2) * 2 * 22)
        self.assertEqual(int(total["duplicated_inferences"]), 2 * 22)   # s2 ran twice, both models
        self.assertEqual(total["exit_code"], "1")


class TestRunConfigRecord(unittest.TestCase):
    def test_settings_without_bucket_and_clip_list_summarised(self):
        env = {"GCS_BUCKET_NAME": "my-private-bucket", "CLIPS": "a|none,a|WN_3snr,b|none",
               "OMP_NUM_THREADS": "2", "BATCH_SIZE": "22"}
        with mock.patch.dict(os.environ, env):
            rec = utility.run_config_record(Config())
        line = json.dumps(rec)
        self.assertNotIn("my-private-bucket", line)
        self.assertEqual(rec["config"]["CLIPS"], {"samples": 2, "clips": 3})
        self.assertEqual(rec["config"]["OMP_NUM_THREADS"], "2")
        self.assertEqual(rec["config"]["BATCH_SIZE"], 22)
        self.assertIn("WHISPER_FALLBACK", rec["config"])
        self.assertIn("python", rec["versions"])

    def test_does_not_import_torch(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            rec = utility.run_config_record(Config())
            self.assertNotIn("torch", sys.modules)
        self.assertIsNone(rec["gpu"])

    def test_print_timing_line(self):
        _, recs = capture_timing(utility.print_timing, {"event": "phase", "name": "x", "seconds": 1})
        self.assertEqual(recs[0]["name"], "x")
        self.assertIn("ts", recs[0])


class TestProgress(unittest.TestCase):
    def test_counts_resumed_samples(self):
        log = textwrap.dedent("""\
            Initializing HF pipeline: a
            Initializing HF pipeline: b
            ===== Run Batch Checkpoint Inference on 100 samples =====
            Resuming pipeline from checkpoint 0001 (sample 50)...
            === Checkpoint #1 ===
            """)
        log += "".join(f'[TIMING] {{"ts": "2026-10-29T01:00:{i:02d}+00:00", "event": "sample", '
                       f'"sample_idx": "s{i}"}}\n' for i in range(10))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "container.log"
            p.write_text(log)
            out = subprocess.run(["bash", str(REPO / "progress.sh"), str(p)], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("samples:    60/100", out.stdout)
        self.assertIn("models:     2", out.stdout)


if __name__ == "__main__":
    unittest.main()
