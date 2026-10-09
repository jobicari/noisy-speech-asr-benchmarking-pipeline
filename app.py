from asr_pipeline_benchmark.configuration import Config
import pandas as pd
config = Config()

from asr_pipeline_benchmark.storage import LocalStorage, GCSStorage
from asr_pipeline_benchmark.models import load_transcriber_engine, get_model_backend, run_batch_inference
from asr_pipeline_benchmark.data import get_mit_ir_dataset, get_speech_dataset
from asr_pipeline_benchmark.preprocessing import (
    cleanup_pipeline, add_white_noise, add_synthetic_reverberation,
    extract_ir_from_mit_dataset, extract_irs_from_mit_dataset_by_location,
    add_real_reverberation, add_smart_audio_dropouts,
    packet_loss, band_limit, hard_clip_ratio, hard_clip_ratio_norm,
    codec_degrade, CorruptionListShort, CorruptionListFull
)
from asr_pipeline_benchmark.utility import generate_run_label



if __name__ == "__main__":
    dataset = get_speech_dataset()
    ir_ds = get_mit_ir_dataset()

    # Load storage interface accoring to the config
    if config.USE_CLOUD_STORAGE:
        storageInterface = GCSStorage(config.GCS_BUCKET_NAME)
    else:
        storageInterface = LocalStorage()

    # Determine the right corruptions list depending on the run levels: DR small, DR full, Main run
    if config.TEST_TYPE == "smallDr":
        corruptions_list = CorruptionListShort()
    elif config.TEST_TYPE == "fullDr" or config.TEST_TYPE == "fullRun":
        corruptions_list = CorruptionListFull()
    # Failsafe in case of typo
    else:
        corruptions_list = CorruptionListShort()

    smallModel = None
    largeModel = None

    # Run transcription pipeline
    for run_small_model in [True, False]:
        # SETUP
        transcriber = load_transcriber_engine(config.IS_APPLE_SILICON, model_small=run_small_model)
        testtype = config.TEST_TYPE
        n = config.NUMBER_OF_SAMPLES
        run_label = generate_run_label(testtype, get_model_backend(transcriber), n, MODEL_SMALL=run_small_model)

        # Load iterator on the dataset in streaming mode
        iterator = iter(dataset)

        # RUN
        run_batch_inference(
            transcriber,
            iterator,
            run_label,
            n,
            storageInterface,
            corruption_type=corruptions_list.corruptions,
            clear_cache=True
        )

        # Verification that the samples used in both as the same
        if run_small_model:
            smallModel = storageInterface.load_pickle(
                str(config.TRANSCRIBE_CACHE_DIR) + f"/{run_label}.pkl"
            )
        else:
            largeModel = storageInterface.load_pickle(
                str(config.TRANSCRIBE_CACHE_DIR) + f"/{run_label}.pkl"
            )

    df_smallModel = pd.DataFrame(smallModel)
    df_largeModel = pd.DataFrame(largeModel)

    assert df_smallModel["idx"].equals(df_largeModel["idx"]), "IDX are not the same"
    assert df_smallModel["ref"].equals(df_largeModel["ref"]), "REF are not the same"

    print(f"RUN COMPLETED SUCCESSFULLY")
