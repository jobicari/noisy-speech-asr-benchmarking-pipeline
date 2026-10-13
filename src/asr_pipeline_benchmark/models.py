import pickle
import time
from pathlib import Path
from itertools import islice

from asr_pipeline_benchmark.configuration import Config
from asr_pipeline_benchmark.utility import checkpoint_iterator, print_timing
from asr_pipeline_benchmark.storage import StorageBackend
from asr_pipeline_benchmark.preprocessing import (
    cleanup_pipeline,
    add_white_noise,
    add_synthetic_reverberation,
    add_smart_audio_dropouts,
    add_real_reverberation,
    extract_ir_from_mit_dataset
)
from asr_pipeline_benchmark.audio_analysis import analyze_asr_audio_health

import jiwer

config = Config()


def load_transcriber_engine(IS_APPLE_SILICON=False, FORCE_MPS=False, model_small=config.MODEL_SMALL):
    """
    Automatically detect if Apple Silicon and tries to load mlx_whisper if available
    If not available, falls back to HuggingFace pipeline and setup device accordingly ("cuda", "mps", "cpu")

    Needs GLOBAL VAR: IS_APPLE_SILICON  to be set True or False
    :return:
        if MLX is used returns the mlx transcriber, otherwise
    """
    if IS_APPLE_SILICON:
        try:
            # from mlx_whisper.load_models import load_model as mlx_load_model
            from mlx_whisper import transcribe as mlx_transcribe
            print("Apple Silicon Detected. Initializing MLX.")
            # Using MLX community model
            if model_small:
                model_to_load = config.MLX_SMALL_MODEL_NAME
            else:
                model_to_load = config.MLX_LARGE_MODEL_NAME

            def mlx_transcriber(audio_array_list, batch_size=1):

                results = []
                for audio_array in audio_array_list:
                    sample_result = mlx_transcribe(audio_array, path_or_hf_repo=model_to_load)
                    results.append({"text": sample_result["text"]})
                return results

            mlx_transcriber.model_name = model_to_load
            return mlx_transcriber

        except ImportError:
            print(f"Apple Silicon detected but 'mlx_whisper' not installed. Falling back to PyTorch.")

    # 2 Standard Production/Linux Fallback
    from transformers import pipeline, AutoTokenizer, GenerationConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else ("mps" if (IS_APPLE_SILICON or FORCE_MPS) else "cpu")
    print(f"Standard architecture detected. Initializing Hugging Face pipeline on device: {device}")
    dtype = torch.float32 if device=="mps" else torch.float16

    if model_small:
        model_id = config.HF_TRANSFORMER_SMALL_MODEL_NAME
    else:
        model_id = config.HF_TRANSFORMER_LARGE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)

    whisper_config = GenerationConfig.from_pretrained(model_id)
    whisper_config.task = "transcribe"
    whisper_config.language = "en"
    # whisper_config.num_beams = 1

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device
    )

    # Enclose pipeline in a wrapper function to pass the manual config along
    def hf_transcriber(audio_array_list, batch_size=1):
        result = whisper_pipe(audio_array_list, generation_config=whisper_config, batch_size=batch_size)
        # print(f"Transcriber engine initialized. ({model_id})")
        return result

    hf_transcriber.model_name = whisper_pipe.model.name_or_path
    return hf_transcriber




def get_model_backend(model) -> str:
    """Helper Function that returns 'mlx' or 'torch' based on the instantiated model object"""
    module_name = getattr(model, "__name__", "")

    if "mlx" in module_name:
        return "mlx"
    elif "torch" in module_name or "transformers" or"hf_" in module_name:
        return "torch"
    else:
        return f"unknown ({module_name})"



# =================================================================
# NEW VERSION
# =================================================================

def run_benchmark_test(transcriber, cp_batch, n_samples=10, corruption_type=None):   # no noise
    """Function that runs the inference loop on a checkpoint batch of audio samples
        with a list of corruption type and run error metrics (WER, CER)
        and return the results
    """
    print(f" === RUN BENCHMARK TEST started ===")
    if corruption_type is None:
        corruption_type = ["none"]

    # This should be moved to configuration and not hard coded
    target_snr = 3  # signal-to-noise ratio in dB to be added to the samples

    results = []

    for idx, sample in enumerate(cp_batch):

        # -----------------------------------------------------------------------------
        # Perform Preprocessing (Corruptions and Sample Analysis)
        # -----------------------------------------------------------------------------

        # Start time for the CPU bound prep process
        timer_prep_start = time.perf_counter()

        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]

        corr_audio_list = []
        metadata = []

        # Compute ref once per sample
        ref = cleanup_pipeline(sample["text"])

        for corr_type, corr_fn in corruption_type.items():
            # counter+=1

            corr_audio_array = corr_fn(audio_array)

            # Build the list of raw audio file to send to the transcriber engine
            corr_audio_list.append(corr_audio_array)

            # Run audio health analysis
            sample_analysis = analyze_asr_audio_health(corr_audio_array, sample_rate)
            metadata.append({"idx": sample['id'],
                             "corr_type": corr_type,
                             "sample_analysis": sample_analysis})


        timer_prep_stop = time.perf_counter()
        t_prep_s = timer_prep_stop - timer_prep_start


        # -----------------------------------------------------------------------------
        # Perform transcriptions
        # -----------------------------------------------------------------------------

        # Start time for the GPU bound inference processes
        timer_infer_start = time.perf_counter()

        transcriptions = transcriber(corr_audio_list, batch_size=config.BATCH_SIZE)

        timer_infer_stop = time.perf_counter()
        t_infer_s = timer_infer_stop - timer_infer_start

        # -----------------------------------------------------------------------------
        # iterate over pipe output to retrieve the last metrics needed
        # -----------------------------------------------------------------------------
        # Start timer for the score process
        timer_score_start = time.perf_counter()
        for out, meta in zip(transcriptions, metadata):

            hyp = cleanup_pipeline(out['text'])
            # -----------------------------------------------------------------------------
            # Word Level metrics - WER
            # -----------------------------------------------------------------------------
            wer_metrics = jiwer.process_words(ref, hyp)
            wer = wer_metrics.wer
            w_hits = wer_metrics.hits
            w_subs = wer_metrics.substitutions
            w_dels = wer_metrics.deletions
            w_ins = wer_metrics.insertions
            num_words = w_hits + w_subs + w_dels


            # -----------------------------------------------------------------------------
            # Characeter Level metrics - CER
            # -----------------------------------------------------------------------------
            char_metrics = jiwer.process_characters(ref, hyp)
            cer = char_metrics.cer
            c_hits = char_metrics.hits
            c_subs = char_metrics.substitutions
            c_dels = char_metrics.deletions
            c_ins = char_metrics.insertions
            num_chars = c_hits + c_subs + c_dels

            results.append({"idx": meta["idx"],
                            "corr_type": meta["corr_type"],
                            "wer": wer,
                            "w_hits": w_hits,
                            "w_subs": w_subs,
                            "w_dels": w_dels,
                            "w_ins": w_ins,
                            "num_words": num_words,
                            "cer": cer,
                            "c_hits": c_hits,
                            "c_subs": c_subs,
                            "c_dels": c_dels,
                            "c_ins": c_ins,
                            "num_chars": num_chars,
                            "sample_analysis": meta["sample_analysis"],
                            "ref": ref,
                            "hyp": hyp})

        timer_score_stop = time.perf_counter()
        t_score_s = timer_score_stop - timer_score_start

        # -----------------------------------------------------------------------------
        # Log sample stats records
        # -----------------------------------------------------------------------------
        n_items = len(corruption_type.items())
        audio_s = audio_array.shape[0]/sample_rate*n_items
        record = {
            "event": "sample",
            "model": transcriber.model_name.replace("/", "_"),
            "sample_idx": sample['id'],
            "n_items": n_items,
            "audio_s": audio_s,
            "t_prep_s": t_prep_s,
            "t_infer_s": t_infer_s,
            "t_score_s": t_score_s
        }
        print_timing(record)


    print(f" === RUN BENCHMARK TEST finished ===")
    return results


def run_checkpoint_inference(
        transcriber,
        iterator,
        run_label,
        n,
        storage: StorageBackend,
        corruption_type: dict =None,
        clear_cache=False
) -> list[dict]:
    """
    Run batch inference per checkpoints with the given transcriber function.

    Parameters:
    -----------
    transcriber : callable
        A function that takes an audio array and returns {"text": str}
    iterator : iterator
        Streaming dataset iterator
    run_label : str
        Unique label for caching
    n : int
        Number of samples to process
    storage : StorageBackend
    corruption_type : dict, optional
        Dict of corruption types and their associated functions
    clear_cache : bool, default=False
        Whether to clear batch checkpoints after assembling final results

    Returns:
        list of dictionaries containing the results of the inference
        every item (dict) in the list corresponds to a single sample
    """
    print(f"===== Run Batch Checkpoint Inference on {n} samples =====")
    print(f"run_label: {run_label}")

    if corruption_type is None:
        corruption_type = ["none"]

    run_file_key = f"{config.TRANSCRIBE_CACHE_DIR}/{run_label}.pkl"
    checkpoint_prefix = f"{config.CHECKPOINT_DIR}/{run_label}_CP_"

    results: list[dict] = []
    cache_loaded_successfully = False

    if storage.exists(run_file_key) and not config.FORCE_RETRAIN:
        try:
            results = storage.load_pickle(run_file_key)
            cache_loaded_successfully = True
            print(f"Loaded test results from cache: \n\t{run_file_key}")

        except Exception as e:
            print(f"Warning: Cache file found but could not be loaded due to error: {e}")

    # Run inference
    if not cache_loaded_successfully and config.RUN_BENCHMARK_INFERENCE:
        sample_counter = 0
        checkpoint_idx = 0
        limited_iterator = islice(iterator, n)

        # Retrieve files via interface and sort by the numeric checkpoint index
        existing_checkpoints = storage.list_checkpoints(checkpoint_prefix)
        existing_checkpoints = sorted(
            existing_checkpoints,
            key= lambda f: int(Path(f).stem.split("_")[-1]),
        )

        if existing_checkpoints:
            cp_indices = [int(Path(f).stem.split("_")[-1]) for f in existing_checkpoints]
            # Only resume when the checkpoints form a contiguous run starting at 0,
            # otherwise the fast-forward offset is wrong and samples get skipped.
            expected = list(range(max(cp_indices)+1))
            if cp_indices == expected:
                checkpoint_idx = max(cp_indices) + 1
                sample_counter = checkpoint_idx * config.CHECKPOINT_SIZE
                print(f"Resuming pipeline from checkpoint {checkpoint_idx:04d} (sample {sample_counter})...")

                # fast-forward the lazy iterator
                for _ in range(sample_counter):
                    next(limited_iterator, None)

            else:
                print("Non-contiguous checkpoints found; ignoring resume to keep sampling deterministic.")

        # Generate and process checkpoints
        for cp_batch in checkpoint_iterator(limited_iterator, checkpoint_size=config.CHECKPOINT_SIZE):
            print(f"\n=== Checkpoint #{checkpoint_idx} ===")
            cp_file_key = f"{checkpoint_prefix}{checkpoint_idx:04d}.pkl"

            # === EXECUTION OF THE TRANSCRIPTION TASK ===
            checkpoint_results = run_benchmark_test(transcriber, cp_batch, n_samples=n, corruption_type=corruption_type)

            timer_checkpoints_start = time.perf_counter()

            storage.save_pickle(checkpoint_results, cp_file_key)
            print(f"Successfully dumped checkpoint {checkpoint_idx} results to cache: \n\t{cp_file_key}")

            timer_checkpoints_stop = time.perf_counter()
            t_chkpnt_s = timer_checkpoints_stop - timer_checkpoints_start

            record = {
                "event": "checkpoint",
                "checkpoint_idx": checkpoint_idx,
                "model": transcriber.model_name.replace("/", "_"),
                "seconds": t_chkpnt_s
            }
            print_timing(record)

            checkpoint_idx += 1
            sample_counter += len(cp_batch)


        # === BUILD ONE FINAL DATA FILE ===
        # makes sure files are sorted on checkpoint indices

        timer_finalize_start = time.perf_counter()

        checkpoint_files = sorted(
            storage.list_checkpoints(checkpoint_prefix),
            key= lambda f: int(Path(f).stem.split("_")[-1]),
        )

        # Load and concatenate
        for file_key in checkpoint_files:
            cp_data = storage.load_pickle(file_key)
            results.extend(cp_data)

        print(f"Benchmark Inference Tests completed with {n} samples")

        storage.save_pickle(results, run_file_key)
        print(f"Successfully dumped test results to cache: \n\t{run_file_key}")

        if clear_cache and storage.exists(run_file_key):
            for file_key in checkpoint_files:
                storage.delete(file_key)

        timer_finalize_stop = time.perf_counter()
        t_finalize_s = timer_finalize_stop - timer_finalize_start
        record = {
            "event": "phase",
            "name": "finalize",
            "model": transcriber.model_name.replace("/", "_"),
            "seconds": t_finalize_s
        }
        print_timing(record)

    return results
