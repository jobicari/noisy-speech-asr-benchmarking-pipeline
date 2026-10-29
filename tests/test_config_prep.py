"""CLIPS, WHISPER_FALLBACK and the prep worker's clip filter"""
import os
import unittest
from unittest import mock

import numpy as np

import tests  # noqa: F401  (sets env and sys.path)
from tests.helpers import capture_timing, events
from asr_pipeline_benchmark import configuration, prep
from asr_pipeline_benchmark.configuration import Config, _parse_clips


def fake_sample(sample_id, seconds=1.0, sr=16_000):
    rng = np.random.default_rng(abs(hash(sample_id)) % 2**32)
    return {"id": sample_id, "text": "Hello World",
            "audio": {"array": (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32),
                      "sampling_rate": sr}}


class TestParseClips(unittest.TestCase):
    def test_pairs_grouped_by_sample_in_order(self):
        self.assertEqual(_parse_clips("a|WN_3snr, a|none ,b|SV_0.5,"),
                         {"a": ["WN_3snr", "none"], "b": ["SV_0.5"]})

    def test_empty_means_no_filter(self):
        self.assertEqual(_parse_clips(""), {})

    def test_malformed_entry_raises(self):
        for bad in ("a", "a|", "|none", "a-none"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _parse_clips(bad)


class TestConfigFlags(unittest.TestCase):
    def test_defaults_keep_main_run_behaviour(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WHISPER_FALLBACK", None)
            os.environ.pop("CLIPS", None)
            cfg = Config()
        self.assertFalse(cfg.WHISPER_FALLBACK)
        self.assertEqual(cfg.CLIPS, {})
        self.assertEqual(cfg.WHISPER_LONGFORM, "chunked")

    def test_env_values(self):
        with mock.patch.dict(os.environ, {"WHISPER_FALLBACK": "True", "CLIPS": "x|none,y|WN_3snr"}):
            cfg = Config()
        self.assertTrue(cfg.WHISPER_FALLBACK)
        self.assertEqual(cfg.CLIPS, {"x": ["none"], "y": ["WN_3snr"]})


@mock.patch.object(prep, "cleanup_pipeline", str.lower)                     # no Hub download
@mock.patch.object(prep, "analyze_asr_audio_health", lambda a, sr: {"ok": True})
class TestPrepClipFilter(unittest.TestCase):
    def tearDown(self):
        prep.init_worker("smallDr", None)

    def test_unknown_corruption_rejected(self, *_):
        with self.assertRaises(ValueError):
            prep.init_worker("smallDr", None, {"a": ["not_a_corruption"]})

    def test_no_clips_runs_every_corruption(self, *_):
        prep.init_worker("smallDr", None)
        (audio, meta, ref, _), recs = capture_timing(prep.prep_sample, fake_sample("a"))
        self.assertEqual([m["corr_type"] for m in meta], ["none", "WN_3snr", "SV_0.5"])
        self.assertEqual(len(audio), 3)
        self.assertEqual(events(recs, "prep")[0]["n_items"], 3)
        self.assertEqual(ref, "hello world")

    def test_clips_runs_only_listed_corruptions_in_order(self, *_):
        prep.init_worker("smallDr", None, {"a": ["SV_0.5", "none"], "b": ["WN_3snr"]})
        (audio, meta, _, _), recs = capture_timing(prep.prep_sample, fake_sample("a"))
        self.assertEqual([m["corr_type"] for m in meta], ["SV_0.5", "none"])
        self.assertTrue(all(m["idx"] == "a" for m in meta))
        self.assertEqual(events(recs, "prep")[0]["n_items"], 2)

    def test_clip_audio_matches_full_run(self, *_):
        """A rerun of one clip must corrupt the audio exactly as the full run did"""
        sample = fake_sample("6930-75918-0003", seconds=2.0)
        prep.init_worker("smallDr", None)
        (full_audio, full_meta, _, _), _ = capture_timing(prep.prep_sample, sample)
        full = dict(zip((m["corr_type"] for m in full_meta), full_audio))

        prep.init_worker("smallDr", None, {sample["id"]: ["WN_3snr"]})
        (clip_audio, clip_meta, _, _), _ = capture_timing(prep.prep_sample, sample)
        self.assertEqual(clip_meta[0]["corr_type"], "WN_3snr")
        np.testing.assert_array_equal(clip_audio[0], full["WN_3snr"])

    def test_sample_not_in_clips_gets_nothing(self, *_):
        prep.init_worker("smallDr", None, {"other": ["none"]})
        (audio, meta, _, _), _ = capture_timing(prep.prep_sample, fake_sample("a"))
        self.assertEqual((audio, meta), ([], []))


if __name__ == "__main__":
    unittest.main()
