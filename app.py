import os
import time
import multiprocessing as mp
from asr_pipeline_benchmark.configuration import Config
config = Config()


if __name__ == "__main__":
    from asr_pipeline_benchmark.utility import generate_run_label, print_timing
    from asr_pipeline_benchmark.models import load_transcriber_engine, run_checkpoint_loop
    from asr_pipeline_benchmark.storage import LocalStorage, GCSStorage
    from asr_pipeline_benchmark.data import get_speech_dataset
    from asr_pipeline_benchmark.preprocessing import extract_ir_from_mit_dataset
    from asr_pipeline_benchmark.prep import init_worker

    dataset = get_speech_dataset()
    mit_ir, _ = extract_ir_from_mit_dataset(ir_idx=251)

    # ----------------------------------------------------------------------------------------
    # Load storage interface accoring to the config
    # ----------------------------------------------------------------------------------------
    if config.USE_CLOUD_STORAGE:
        storageInterface = GCSStorage(config.GCS_BUCKET_NAME)
    else:
        storageInterface = LocalStorage()

    # ----------------------------------------------------------------------------------------
    # SPAWN POOL OF WORKERS
    # ----------------------------------------------------------------------------------------
    ctx = mp.get_context('spawn')
    with ctx.Pool(config.N_PREP_WORKERS, initializer=init_worker, initargs=(config.TEST_TYPE, mit_ir)) as pool:
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
        run_checkpoint_loop(
            transcribers,
            pool,
            iterator,
            run_label,
            config.NUMBER_OF_SAMPLES,
            storageInterface,
            clear_cache=True
        )

    print(f"RUN COMPLETED SUCCESSFULLY", flush=True)

    os._exit(0)
