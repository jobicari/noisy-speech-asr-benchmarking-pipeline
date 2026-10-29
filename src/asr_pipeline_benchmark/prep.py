import numpy as np
import time
from asr_pipeline_benchmark.utility import _seed, print_timing
from asr_pipeline_benchmark.preprocessing import (
    cleanup_pipeline, CorruptionListShort, CorruptionListFull
)
from asr_pipeline_benchmark.audio_analysis import analyze_asr_audio_health


_corruptions = None  # one per worker process
_clips = {}

def init_worker(test_type, ir, clips=None):
    global _corruptions, _clips
    lst = CorruptionListFull(ir) if test_type in ("fullDr", "fullRun") else CorruptionListShort()
    _corruptions = lst.corruptions
    _clips = clips or {}
    unknown = {c for corrs in _clips.values() for c in corrs} - set(_corruptions)
    if unknown:
        raise ValueError(f"CLIPS names corruptions not in the {test_type} list: {sorted(unknown)}")


def _sample_corruptions(sample_id):
    """The corruptions to run on a sample: all of them, or only those listed in CLIPS"""
    if not _clips:
        return _corruptions.items()
    return [(c, _corruptions[c]) for c in _clips.get(sample_id, [])]


def prep_sample(sample):

    # Start time for the CPU bound prep process
    timer_prep_start = time.perf_counter()
    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]

    corr_audio_list = []
    metadata = []

    # Compute ref once per sample
    ref = cleanup_pipeline(sample["text"])
    corruptions = _sample_corruptions(sample['id'])
    n_items = len(corruptions)

    for corr_type, corr_fn in corruptions:
        np.random.seed(_seed( sample['id'], corr_type))
        corr_audio_array = corr_fn(audio_array)
        # Build the list of raw audio file to send to the transcriber engine
        corr_audio_list.append(corr_audio_array)
        # Run audio health analysis
        sample_analysis = analyze_asr_audio_health(corr_audio_array, sample_rate)
        duration_s = len(corr_audio_array)/sample_rate
        metadata.append({"idx": sample['id'],
                         "corr_type": corr_type,
                         "duration_s": duration_s,
                         "sample_analysis": sample_analysis})


    timer_prep_stop = time.perf_counter()
    t_prep_s = timer_prep_stop - timer_prep_start
    record = {
        "event": "prep",
        "sample_idx": sample['id'],
        "n_items": n_items,
        "t_prep_s": t_prep_s
    }
    print_timing(record)

    return corr_audio_list, metadata, ref, t_prep_s