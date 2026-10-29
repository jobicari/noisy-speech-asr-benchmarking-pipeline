"""run_and_terminate.sh and run_sweep.sh end to end, with fake docker, nvidia-smi and
curl (tests/fakebin): detached start, main run, long-clip and looped-clip side runs,
timing extraction, analyze, upload, results copy, notifications and the stall alert.
Nothing is uploaded, terminated or sent."""
import json
import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import tests

REPO = tests.REPO
FAKEBIN = REPO / "tests" / "fakebin"
MODELS = "openai/whisper-tiny,facebook/wav2vec2-large-960h-lv60-self,openai/whisper-base"


class WrapperRun(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.fake = self.tmp / "fake"
        self.fake.mkdir()
        self.out = self.tmp / "out"
        (self.tmp / "env").write_text(f"HF_MODELS={MODELS}\nGCS_BUCKET_NAME=test-bucket\n")
        (self.fake / "long_ids").write_text("s9\n")
        (self.fake / "loop_clips").write_text("s2|WN_3snr,s1|BL_500_1000,s1|none\n")

    def run_wrapper(self, sweep="t1", detach="0", extra_docker="", **env):
        e = {**os.environ,
             "PATH": f"{FAKEBIN}:{os.environ['PATH']}", "FAKE_DIR": str(self.fake),
             "SWEEP_ID": sweep, "OUT_ROOT": str(self.out), "ENV_FILE": str(self.tmp / "env"),
             "KEY_FILE": str(self.tmp / "key.json"), "IMAGE": "asr:test", "TERMINATE": "0",
             "DETACH": detach, "NTFY_TOPIC": "test-topic", "CONFIGS": "SAMPLES_PER_CALL=1",
             "PER_RUN_ENV": f"RUN_POSTFIX={sweep}_{{run_id}},NUMBER_OF_SAMPLES=4,BATCH_SIZE=22",
             "EXTRA_DOCKER_ARGS": extra_docker, "GPU_SAMPLE_MS": "200", **env}
        return subprocess.run(["./run_and_terminate.sh"], cwd=REPO, env=e,
                              capture_output=True, text=True, timeout=180)

    def pipeline_log(self, sweep="t1"):
        return (self.out / f"{sweep}.pipeline.log").read_text()

    def container_env(self, sweep):
        """The -e values the run of a sweep got, last one winning"""
        envs = {}
        for f in (self.fake / "containers").glob("*.env"):
            postfix = next(l for l in reversed(f.read_text().splitlines()) if l.startswith("RUN_POSTFIX="))
            run_env = {}
            for line in f.read_text().splitlines():
                k, _, v = line.partition("=")
                run_env[k] = v
            envs[postfix.split("=", 1)[1]] = run_env
        return envs

    def calls(self):
        p = self.fake / "calls.log"
        return p.read_text().splitlines() if p.exists() else []

    def notifications(self):
        p = self.fake / "curl.log"
        return re.findall(r"Title: (\S+: [^-]+?) -d", p.read_text()) if p.exists() else []


class TestFullFlow(WrapperRun):
    def test_main_and_both_side_runs(self):
        res = self.run_wrapper()
        log = self.pipeline_log()
        self.assertEqual(res.returncode, 0, log)
        self.assertIn("Main sweep succeeded", log)
        self.assertIn("Side run t1_longform: 1 samples, models openai/whisper-tiny,openai/whisper-base", log)
        self.assertIn("Loop side run t1_loops: 3 clips in 2 samples, models openai/whisper-tiny,openai/whisper-base", log)
        self.assertIn("Done (TERMINATE=0). main=ok side=1 loops=1", log)

        envs = self.container_env("t1")
        self.assertEqual(set(envs), {"t1_SAMPLES_PER_CALL_1_r1", "t1_longform", "t1_loops"})
        main, loops = envs["t1_SAMPLES_PER_CALL_1_r1"], envs["t1_loops"]
        self.assertNotIn("WHISPER_FALLBACK", main)
        self.assertNotIn("CLIPS", main)
        self.assertEqual(loops["WHISPER_FALLBACK"], "True")
        self.assertEqual(loops["CLIPS"], "s1|BL_500_1000,s1|none,s2|WN_3snr")
        self.assertEqual(loops["NUMBER_OF_SAMPLES"], "2")
        self.assertEqual(loops["HF_MODELS"], "openai/whisper-tiny,openai/whisper-base")
        self.assertEqual(loops["SAMPLES_PER_CALL"], "20")
        self.assertEqual(loops["BATCH_SIZE"], "22")
        self.assertEqual(envs["t1_longform"]["WHISPER_LONGFORM"], "sequential")

        # Sweep folders: provenance, timing, analysis
        for sweep in ("t1", "t1_longform", "t1_loops"):
            sj = json.loads((self.out / sweep / "sweep.json").read_text())
            self.assertRegex(sj["git_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(sj["image_id"], "sha256:fakeimage")
            self.assertEqual(sj["image_revision"], "unknown")   # image built without the label
            self.assertIn(sj["git_dirty"], (True, False))
            run = next(p for p in (self.out / sweep).iterdir() if (p / "run.json").exists())
            self.assertEqual(json.loads((run / "run.json").read_text())["exit_code"], 0)
            timing = (run / "timing.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(t)["event"] for t in timing], ["phase", "sample"])
            self.assertTrue((run / "gpu.csv").stat().st_size > 0)
            self.assertTrue((self.out / sweep / "report.md").exists(), sweep)

        # Uploads, then results copied next to each run's logs
        calls = self.calls()
        uploads = [c.split()[2] for c in calls if c.startswith("upload_results.py")]
        self.assertEqual(uploads, ["lambda_bench_runs/t1", "lambda_bench_runs/t1_longform",
                                   "lambda_bench_runs/t1_loops"])
        copies = [c.split()[1:] for c in calls if c.startswith("copy_results.py")]
        self.assertIn(["results/transcribe_cache/fullRun__hf__x__t1_SAMPLES_PER_CALL_1_r1.pkl",
                       "lambda_bench_runs/t1/SAMPLES_PER_CALL_1_r1"], copies)
        self.assertIn(["results/transcribe_cache/fullRun__hf__x__t1_loops.pkl",
                       "lambda_bench_runs/t1_loops/WHISPER_FALLBACK_True_r1"], copies)
        self.assertEqual(len(copies), 3)
        self.assertTrue(any(c.startswith("find_loop_clips.py results/transcribe_cache/fullRun__hf__x__t1_"
                                         "SAMPLES_PER_CALL_1_r1.pkl --min-ratio 2.4") for c in calls))

        notes = self.notifications()
        self.assertEqual([n.split(": ", 1)[1] for n in notes], ["started", "done"])

    def test_sweep_records_the_image_revision_label(self):
        (self.fake / "image_rev").write_text("0123abcd" * 5 + "\n")
        (self.fake / "loop_clips").write_text("\n")
        (self.fake / "long_ids").write_text("\n")
        self.assertEqual(self.run_wrapper().returncode, 0, self.pipeline_log())
        sj = json.loads((self.out / "t1" / "sweep.json").read_text())
        self.assertEqual(sj["image_revision"], "0123abcd" * 5)

    def test_no_loops_skips_the_loop_run(self):
        (self.fake / "loop_clips").write_text("\n")
        res = self.run_wrapper()
        log = self.pipeline_log()
        self.assertEqual(res.returncode, 0, log)
        self.assertIn("No looped transcripts: skipping the loop side run", log)
        self.assertIn("loops=-", log)
        self.assertFalse((self.out / "t1_loops").exists())

    def test_failed_main_run_skips_side_runs_and_copies(self):
        res = self.run_wrapper(extra_docker="-e FAIL_ME=1")
        log = self.pipeline_log()
        self.assertIn("FAILED", log)
        self.assertIn("main=FAILED side=- loops=-", log)
        self.assertFalse(any(c.startswith(("copy_results.py", "find_")) for c in self.calls()))
        self.assertIn("MAIN RUN FAILED", " ".join(self.notifications()))
        self.assertEqual(res.returncode, 0)   # TERMINATE=0 ends normally after the upload

    def test_failed_results_copy_keeps_instance_and_alerts(self):
        (self.fake / "copy_exit").write_text("1\n")
        res = self.run_wrapper()
        log = self.pipeline_log()
        self.assertEqual(res.returncode, 1)
        self.assertIn("RESULTS COPY NOT VERIFIED", log)
        self.assertIn("RESULTS COPY FAILED", " ".join(self.notifications()))
        self.assertNotIn("Done (TERMINATE", log)

    def test_detached_run_returns_at_once_and_finishes(self):
        t0 = time.time()
        res = self.run_wrapper(detach="1")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Running detached", res.stdout)
        self.assertLess(time.time() - t0, 10)
        log_path = self.out / "t1.pipeline.log"
        deadline = time.time() + 150
        while time.time() < deadline and "Done (TERMINATE=0)" not in (log_path.read_text() if log_path.exists() else ""):
            time.sleep(1)
        self.assertIn("main=ok side=1 loops=1", log_path.read_text())


class TestScriptModes(unittest.TestCase):
    def test_shell_scripts_are_executable_in_git(self):
        """A fresh clone must be able to run ./run_and_terminate.sh and ./run_sweep.sh"""
        out = subprocess.run(["git", "ls-files", "-s", "*.sh"], cwd=REPO, capture_output=True, text=True)
        modes = dict((l.split()[3], l.split()[0]) for l in out.stdout.splitlines())
        self.assertGreaterEqual(set(modes), {"run_and_terminate.sh", "run_sweep.sh", "progress.sh"})
        for name, mode in modes.items():
            self.assertEqual(mode, "100755", name)


def extract_function(name):
    """The source of one shell function in run_and_terminate.sh"""
    text = (REPO / "run_and_terminate.sh").read_text()
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0)


class TestStallAlert(unittest.TestCase):
    def test_alerts_once_per_stall(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            log = d / "out/sw_loops/run1/container.log"
            log.parent.mkdir(parents=True)
            log.write_text("x\n")
            os.utime(log, (time.time() - 3600, time.time() - 3600))
            script = d / "t.sh"
            script.write_text(
                'log() { echo "LOG $*"; }\nnotify() { echo "NOTIFY $1"; }\n'
                + extract_function("watch_stall").replace("sleep 60", "sleep 0.3")
                + f'OUT_ROOT={d}/out SWEEP_ID=sw STALL_MIN=30\nwatch_stall & p=$!\n'
                  f'sleep 1.2; touch {log}; sleep 1; touch -t 202601010000 {log}; sleep 1\nkill $p\n')
            out = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30,
                                 env={**os.environ, "PATH": f"{FAKEBIN}:{os.environ['PATH']}"})
        notes = [l for l in out.stdout.splitlines() if l.startswith("NOTIFY")]
        self.assertEqual(len(notes), 2, out.stdout + out.stderr)   # stale, fresh (reset), stale again
        self.assertTrue(all(n.startswith("NOTIFY no progress for") for n in notes))

    def test_quiet_while_logs_grow(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            log = d / "out/sw/run1/container.log"
            log.parent.mkdir(parents=True)
            log.write_text("x\n")
            script = d / "t.sh"
            script.write_text(
                'log() { echo "LOG $*"; }\nnotify() { echo "NOTIFY $1"; }\n'
                + extract_function("watch_stall").replace("sleep 60", "sleep 0.3")
                + f'OUT_ROOT={d}/out SWEEP_ID=sw STALL_MIN=30\nwatch_stall & p=$!\nsleep 1.5\nkill $p\n')
            out = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30,
                                 env={**os.environ, "PATH": f"{FAKEBIN}:{os.environ['PATH']}"})
        self.assertNotIn("NOTIFY", out.stdout)


if __name__ == "__main__":
    unittest.main()
