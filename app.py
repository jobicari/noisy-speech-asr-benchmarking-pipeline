import os
import time
from asr_pipeline_benchmark.configuration import Config
import pandas as pd
config = Config()

from asr_pipeline_benchmark.storage import LocalStorage, GCSStorage
from asr_pipeline_benchmark.models import load_transcriber_engine, run_checkpoint_inference
from asr_pipeline_benchmark.data import get_mit_ir_dataset, get_speech_dataset
from asr_pipeline_benchmark.preprocessing import (
    cleanup_pipeline, add_white_noise, add_synthetic_reverberation,
    extract_ir_from_mit_dataset, extract_irs_from_mit_dataset_by_location,
    add_real_reverberation, add_smart_audio_dropouts,
    packet_loss, band_limit, hard_clip_ratio, hard_clip_ratio_norm,
    codec_degrade, CorruptionListShort, CorruptionListFull
)
from asr_pipeline_benchmark.utility import generate_run_label, print_timing



if __name__ == "__main__":
    dataset = get_speech_dataset()
    ir_ds = get_mit_ir_dataset()

    # ----------------------------------------------------------------------------------------
    # Load storage interface accoring to the config
    # ----------------------------------------------------------------------------------------
    if config.USE_CLOUD_STORAGE:
        storageInterface = GCSStorage(config.GCS_BUCKET_NAME)
    else:
        storageInterface = LocalStorage()

    # ----------------------------------------------------------------------------------------
    # Determine the right corruptions list depending on the run levels: DR small, DR full, Main run
    # ----------------------------------------------------------------------------------------
    if config.TEST_TYPE == "smallDr":
        corruptions_list = CorruptionListShort()
    elif config.TEST_TYPE == "fullDr" or config.TEST_TYPE == "fullRun":
        corruptions_list = CorruptionListFull()
    # Failsafe in case of typo
    else:
        corruptions_list = CorruptionListShort()

    smallModel = None
    largeModel = None


    # ----------------------------------------------------------------------------------------
    # LOAD TRANSCRIBER ENGINE
    # ----------------------------------------------------------------------------------------
    transcribers = []

    for model_id in config.MODELS:

        timer_load_start =  time.perf_counter()
        transcriber = load_transcriber_engine(config.INFERENCE_BACKEND, model_id)
        timer_load_stop = time.perf_counter()
        t_load_s = timer_load_stop - timer_load_start
        record = {
            "event": "phase",
            "name": "model_load",
            "model": transcriber.model_name.replace("/", "_"),
            "seconds": t_load_s
        }
        print_timing(record)

        transcribers.append(transcriber)

    # ----------------------------------------------------------------------------------------
    # CONFIGURATION
    # ----------------------------------------------------------------------------------------
    run_label = generate_run_label(
        config.TEST_TYPE,
        config.INFERENCE_BACKEND,
        config.MODELS,
        config.NUMBER_OF_SAMPLES,
        postfix=config.RUN_POSTFIX
    )
    # Load iterator on the dataset in streaming mode
    iterator = iter(dataset)
    # ----------------------------------------------------------------------------------------
    # RUN INFERENCE
    # ----------------------------------------------------------------------------------------
    run_checkpoint_inference(
        transcribers,
        iterator,
        run_label,
        config.NUMBER_OF_SAMPLES,
        storageInterface,
        corruption_type=corruptions_list.corruptions,
        clear_cache=True
    )
    iterator.close()

    print(f"RUN COMPLETED SUCCESSFULLY", flush=True)

    os._exit(0)
