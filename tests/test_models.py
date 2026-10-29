"""run_checkpoint_loop timing, checkpoints and resume, and the Whisper loader's
fallback switch. Fake models and an in-process pool stand in for the GPU and workers."""
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tests  # noqa: F401
from tests.helpers import capture_timing, events
from tests.test_config_prep import fake_sample
from asr_pipeline_benchmark import models, prep
from asr_pipeline_benchmark.storage import LocalStorage

N_CORR = 3  # smallDr


class InlinePool:
    """multiprocessing.Pool stand-in: imap runs lazily in this process"""
    def imap(self, fn, it):
        return map(fn, it)


class FakeModel:
    def __init__(self, name, text="hello world"):
        self.model_name, self.text, self.calls = name, text, []

    def __call__(self, audio_list, batch_size=1):
        self.calls.append((len(audio_list), batch_size))
        return [{"text": self.text} for _ in audio_list]


@mock.patch.object(prep, "cleanup_pipeline", str.lower)
@mock.patch.object(prep, "analyze_asr_audio_health", lambda a, sr: {})
@mock.patch.object(models, "cleanup_pipeline", str.lower)
class TestCheckpointLoop(unittest.TestCase):
    def setUp(self):
        prep.init_worker("smallDr", None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        patcher = mock.patch.multiple(
            models.config, TRANSCRIBE_CACHE_DIR=str(self.tmp / "cache"),
            CHECKPOINT_DIR=str(self.tmp / "cache/checkpoint"), CHECKPOINT_SIZE=2,
            BATCH_SIZE=22, SAMPLES_PER_CALL=1, FORCE_RETRAIN=False, RUN_BENCHMARK_INFERENCE=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.storage = LocalStorage()
        self.samples = [fake_sample(f"s{i}") for i in range(5)]
        self.models = [FakeModel("org/tiny"), FakeModel("org/base")]

    def run_loop(self, label="run", n=5):
        return capture_timing(models.run_checkpoint_loop, self.models, InlinePool(),
                              iter(self.samples), label, n, self.storage, clear_cache=True)

    def test_rows_and_timing_records(self):
        rows, recs = self.run_loop()
        self.assertEqual(len(rows), 5 * N_CORR * 2)
        self.assertEqual({(r["idx"], r["model"]) for r in rows},
                         {(f"s{i}", m) for i in range(5) for m in ("org_tiny", "org_base")})

        self.assertEqual(len(events(recs, "prep")), 5)
        samples = events(recs, "sample")
        self.assertEqual(len(samples), 5 * 2)  # one per sample per model
        for s in samples:
            for key in ("model", "sample_idx", "n_items", "audio_s", "t_infer_s", "t_score_s",
                        "t_wait_s", "group_size", "ts"):
                self.assertIn(key, s)
            self.assertEqual(s["n_items"], N_CORR)
            self.assertAlmostEqual(s["audio_s"], N_CORR * 1.0, places=3)
        ckpts = events(recs, "checkpoint")
        self.assertEqual([c["checkpoint_idx"] for c in ckpts], [0, 1, 2])
        self.assertTrue(all("t_fetch_s" in c and "seconds" in c for c in ckpts))
        self.assertEqual(len(events(recs, "phase", "finalize")), 1)
        self.assertEqual(events(recs, "phase", "resume_skip"), [])

    def test_final_file_written_and_checkpoints_cleared(self):
        self.run_loop()
        self.assertTrue((self.tmp / "cache/run.pkl").exists())
        self.assertEqual(list((self.tmp / "cache/checkpoint").glob("*.pkl")), [])

    def test_every_model_gets_each_sample_in_one_batched_call(self):
        self.run_loop()
        for m in self.models:
            self.assertEqual(m.calls, [(N_CORR, 22)] * 5)

    def test_resume_skips_saved_checkpoints_and_logs_it(self):
        saved = [{"idx": "s0", "model": "saved"}, {"idx": "s1", "model": "saved"}]
        self.storage.save_pickle(saved, str(self.tmp / "cache/checkpoint/run_CP_0000.pkl"))

        rows, recs = self.run_loop()
        skip = events(recs, "phase", "resume_skip")
        self.assertEqual(len(skip), 1)
        self.assertEqual((skip[0]["checkpoint_idx"], skip[0]["samples_skipped"]), (1, 2))
        self.assertIn("seconds", skip[0])
        # Only s2..s4 were transcribed, checkpoints continue at 1
        self.assertEqual({s["sample_idx"] for s in events(recs, "sample")}, {"s2", "s3", "s4"})
        self.assertEqual([c["checkpoint_idx"] for c in events(recs, "checkpoint")], [1, 2])
        self.assertEqual(rows[:2], saved)
        self.assertEqual(len(rows), 2 + 3 * N_CORR * 2)

    def test_non_contiguous_checkpoints_do_not_resume(self):
        self.storage.save_pickle([], str(self.tmp / "cache/checkpoint/run_CP_0001.pkl"))
        _, recs = self.run_loop()
        self.assertEqual(events(recs, "phase", "resume_skip"), [])
        self.assertEqual(len({s["sample_idx"] for s in events(recs, "sample")}), 5)

    def test_cached_results_are_loaded_without_inference(self):
        self.storage.save_pickle([{"cached": True}], str(self.tmp / "cache/run.pkl"))
        rows, recs = self.run_loop()
        self.assertEqual(rows, [{"cached": True}])
        self.assertEqual(events(recs, "sample"), [])
        self.assertEqual(self.models[0].calls, [])

    def test_samples_per_call_groups_and_charges_wait_once(self):
        models.config.SAMPLES_PER_CALL = 2
        _, recs = self.run_loop()
        # checkpoints of 2, 2, 1 samples -> 3 calls per model
        self.assertEqual([c[0] for c in self.models[0].calls], [2 * N_CORR, 2 * N_CORR, N_CORR])
        tiny = [s for s in events(recs, "sample") if s["model"] == "org_tiny"]
        self.assertEqual([s["group_size"] for s in tiny], [2, 2, 2, 2, 1])
        self.assertEqual([s["t_wait_s"] == 0.0 for s in tiny], [False, True, False, True, False])

    def test_clip_list_rows_only_for_listed_clips(self):
        prep.init_worker("smallDr", None, {"s0": ["WN_3snr"], "s1": ["none", "SV_0.5"]})
        self.samples = self.samples[:2]
        rows, recs = self.run_loop(n=2)
        self.assertEqual(sorted({(r["idx"], r["corr_type"]) for r in rows}),
                         [("s0", "WN_3snr"), ("s1", "SV_0.5"), ("s1", "none")])
        self.assertEqual([s["n_items"] for s in events(recs, "sample") if s["model"] == "org_tiny"], [1, 2])


class TestScore(unittest.TestCase):
    def test_wer_counts_insertions_from_a_loop(self):
        meta = {"idx": "a", "corr_type": "none", "duration_s": 1.0, "sample_analysis": {}}
        with mock.patch.object(models, "cleanup_pipeline", str.lower):
            row = models._score("i am i am i am i am", "i am", meta, "m")
        self.assertEqual((row["w_hits"], row["w_ins"], row["num_words"]), (2, 6, 2))
        self.assertAlmostEqual(row["wer"], 3.0)


def fake_ml_modules(model_type="whisper"):
    """torch and transformers stand-ins that record what the loader asks for"""
    seen = {"pipeline": None, "calls": [], "seed": None}

    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)

    tf = types.ModuleType("transformers")

    class Pipe:
        model = types.SimpleNamespace(name_or_path="org/model")

        def __call__(self, audio, batch_size=1, **kwargs):
            seen["calls"].append(kwargs)
            return [{"text": "x"} for _ in audio]

    def pipeline(task, **kwargs):
        seen["pipeline"] = kwargs
        return Pipe()

    tf.pipeline = pipeline
    tf.AutoConfig = types.SimpleNamespace(from_pretrained=lambda m: types.SimpleNamespace(model_type=model_type))
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda m, **k: "tokenizer")
    tf.GenerationConfig = types.SimpleNamespace(from_pretrained=lambda m: types.SimpleNamespace())
    tf.set_seed = lambda s: seen.__setitem__("seed", s)
    return {"torch": torch, "transformers": tf}, seen


class TestWhisperLoader(unittest.TestCase):
    def load(self, model_type="whisper", **cfg):
        mods, seen = fake_ml_modules(model_type)
        with mock.patch.dict(sys.modules, mods), mock.patch.multiple(models.config, **cfg), \
                contextlib.redirect_stdout(io.StringIO()):
            t = models.load_hf_transcriber("org/model")
            t([np.zeros(10)] * 3, batch_size=22)
        return seen

    def test_main_run_decodes_greedy_and_chunked(self):
        seen = self.load(WHISPER_FALLBACK=False, WHISPER_LONGFORM="chunked")
        self.assertEqual(seen["pipeline"]["chunk_length_s"], 30)
        self.assertEqual(list(seen["calls"][0]), ["generation_config"])
        self.assertIsNone(seen["seed"])

    def test_fallback_uses_openai_defaults_and_seed(self):
        seen = self.load(WHISPER_FALLBACK=True, WHISPER_LONGFORM="chunked", SEED=123)
        call = seen["calls"][0]
        self.assertEqual(call["temperature"], (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertEqual(call["compression_ratio_threshold"], 1.35)
        self.assertEqual(call["logprob_threshold"], -1.0)
        self.assertIn("generation_config", call)
        self.assertEqual(seen["seed"], 123)
        self.assertEqual(seen["pipeline"]["chunk_length_s"], 30)

    def test_fallback_with_sequential_longform(self):
        seen = self.load(WHISPER_FALLBACK=True, WHISPER_LONGFORM="sequential")
        self.assertIsNone(seen["pipeline"]["chunk_length_s"])
        self.assertTrue(seen["calls"][0]["return_timestamps"])
        self.assertIn("temperature", seen["calls"][0])

    def test_ctc_model_ignores_fallback(self):
        seen = self.load(model_type="wav2vec2", WHISPER_FALLBACK=True)
        self.assertEqual(seen["calls"][0], {})


if __name__ == "__main__":
    unittest.main()
