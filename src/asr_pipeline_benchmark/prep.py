import numpy as np
import time
from asr_pipeline_benchmark.utility import _seed, print_timing
from asr_pipeline_benchmark.preprocessing import (
    cleanup_pipeline, CorruptionListShort, CorruptionListFull
)
from asr_pipeline_benchmark.audio_analysis import analyze_asr_audio_health


_corruptions = None  # one per worker process

def init_worker(test_type, ir):
    global _corruptions
    lst = CorruptionListFull(ir) if test_type in ("fullDr", "fullRun") else CorruptionListShort()
    _corruptions = lst.corruptions


def prep_sample(sample):

    # Start time for the CPU bound prep process
    timer_prep_start = time.perf_counter()
    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]

    corr_audio_list = []
    metadata = []

    # Compute ref once per sample
    ref = cleanup_pipeline(sample["text"])
    n_items = len(_corruptions.items())

    for corr_type, corr_fn in _corruptions.items():
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