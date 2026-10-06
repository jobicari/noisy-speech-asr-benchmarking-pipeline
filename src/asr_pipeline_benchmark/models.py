from asr_pipeline_benchmark.configuration import Config
from asr_pipeline_benchmark.utility import batch_iterator
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
import pickle
from itertools import islice

config = Config()


def load_transcriber_engine(IS_APPLE_SILICON=False, FORCE_MPS=False, model_small=config.MODEL_SMALL):
    """
    Automatically detect if Apple Silicon and tries to load mlx_whisper if available
    If not available, falls back to HuggingFace pipeline and setup device accordingly ("cuda", "mps", "cpu")

    Needs GLOBAL VAR: IS_APPLE_SILICON  to be set True or False
    :return:
        if MLX is used returns the mlx transcriber, otherwise
    """
    loaded_model_name = ""
    if IS_APPLE_SILICON:
        try:
            # from mlx_whisper.load_models import load_model as mlx_load_model
            from mlx_whisper import transcribe as mlx_transcribe
            print("Apple Silicon Detected. Initializing MLX.")

            def mlx_transcriber(audio_array):
                # Using MLX community model
                if model_small:
                    result = mlx_transcribe(audio_array, path_or_hf_repo=config.MLX_SMALL_MODEL_NAME)
                    loaded_model_name = config.MLX_SMALL_MODEL_NAME
                    # model = mlx_load_model(config.MLX_SMALL_MODEL_NAME)
                else:
                    result = mlx_transcribe(audio_array, path_or_hf_repo=config.MLX_LARGE_MODEL_NAME)
                    loaded_model_name = config.MLX_LARGE_MODEL_NAME
                    # model = mlx_load_model(config.MLX_LARGE_MODEL_NAME)

                # print(f"Transcriber engine initialized. ({loaded_model_name})")
                return {"text": result["text"]}
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
    def hf_transcriber(audio_array):
        result = whisper_pipe(audio_array, generation_config=whisper_config)
        # print(f"Transcriber engine initialized. ({model_id})")
        return result

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


def run_benchmark_test(transcriber, batch, n_samples=10, corruption_type=None):   # no noise
    """Function that runs the inference loop on a batch of audio samples
        with a list of corruption type and run error metrics (WER, CER)
        and return the results
    """
    print(f" === RUN BENCHMARK TEST started ===")
    if corruption_type is None:
        corruption_type = ["none"]
    # iterator = iter(dataset)
    target_snr = 3  # signal-to-noise ratio in dB to be added to the samples

    counter, running_avg_wer, running_avg_cer = 0, 0, 0
    tot_w_ref, tot_w_hits, tot_w_subs, tot_w_dels, tot_w_ins = 0,0,0,0,0
    tot_c_ref, tot_c_hits, tot_c_subs, tot_c_dels, tot_c_ins = 0,0,0,0,0

    results = []

    for idx, sample in enumerate(batch):
        # sample = next(iterator)
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]

        for corr_type, corr_fn in corruption_type.items():
            counter+=1
            corr_audio_array = corr_fn(audio_array)

            ref = cleanup_pipeline(sample["text"])
            hyp = cleanup_pipeline(transcriber(corr_audio_array)['text'])

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

            # Add to total for macro-averages
            running_avg_wer = (counter-1) / counter*running_avg_wer + wer/counter
            tot_w_ref += num_words
            tot_w_hits += w_hits
            tot_w_subs += w_subs
            tot_w_dels += w_dels
            tot_w_ins += w_ins


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

            # Add to total for macro-averages
            running_avg_cer = (counter-1) / counter*running_avg_cer + cer/counter
            tot_c_ref += num_chars
            tot_c_hits += c_hits
            tot_c_subs += c_subs
            tot_c_dels += c_dels
            tot_c_ins += c_ins

            # -----------------------------------------------------------------------------
            # Store results and references into list for further analysis
            # -----------------------------------------------------------------------------
            sample_analysis = analyze_asr_audio_health(corr_audio_array, sample_rate)

            results.append({"idx": sample['id'],
                            "corr_type": corr_type,
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
                            "sample_analysis": sample_analysis,
                            "ref": ref,
                            "hyp": hyp})



    print(f" === RUN BENCHMARK TEST finished ===")
    return results


def run_batch_inference(transcriber, iterator, run_label, n, corruption_type=None, clear_cache=False):
    """
    Run batch inference with the given transcriber function.

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
    corruption_type : list, optional
        List of corruption types to apply
    clear_cache : bool, default=False
        Whether to clear batch checkpoints after assembling final results
    """
    print(f"===== Run Batch Inference on {n} samples =====")
    print(f"run_label: {run_label}")
    # print(f"corruption_type: {corruption_type}")

    if corruption_type is None:
        corruption_type = ["none"]

    run_filepath = config.TRANSCRIBE_CACHE_PATH / f"{run_label}.pkl"
    results = []
    cache_loaded_successfully = False

    if run_filepath.exists() and not config.FORCE_RETRAIN:
        try:
            with open(run_filepath, "rb") as f:
                results = pickle.load(f)
            cache_loaded_successfully = True
            print(f"Loaded test results from cache: \n\t{run_filepath}")

        except (pickle.UnpicklingError, IOError, EOFError, AttributeError) as e:
            print(f"Warning: Cache file found but could not be loaded due to error: {e}")
            print("Falling back to running the benchmark inference...")

    if not cache_loaded_successfully and config.RUN_BENCHMARK_INFERENCE:

        sample_counter = 0
        batch_idx = 0
        limited_iterator = islice(iterator, n)

        existing_batchs = list(config.CHECKPOINT_DIR.glob(run_label+"*batch_*.pkl"))
        if existing_batchs:
            batch_indices = [int(f.stem.split("_")[-1]) for f in existing_batchs]
            batch_idx = max(batch_indices, default=0) + 1
            sample_counter = batch_idx * config.BATCH_SIZE
            print(f"Resuming pipeline from batch {batch_idx:04d} (approx sample {sample_counter})...")

            # fast-forward the lazy iterator
            for _ in range(sample_counter):
                next(limited_iterator, None)

        for batch in batch_iterator(limited_iterator, batch_size=config.BATCH_SIZE):
            print(f"\n=== Batch #{batch_idx} ===")
            batch_filepath = config.CHECKPOINT_DIR / f"{run_label}_batch_{batch_idx:04d}.pkl"

            batch_results = run_benchmark_test(transcriber, batch, n_samples=n, corruption_type=corruption_type)

            with open(batch_filepath, "wb") as f:
                pickle.dump(batch_results, f)
                print(f"Successfully dumped batch {batch_idx} results to cache: \n\t{batch_filepath}")

            batch_idx += 1
            sample_counter += len(batch)

        # === BUILD ONE FINAL DATA FILE ===
        batch_files = list(config.CHECKPOINT_DIR.glob(run_label+"*batch_*.pkl"))
        print(f"len batch_files: {len(batch_files)}")

        # Load and concatenate
        for file in batch_files:
            with open(file, "rb") as f:
                batch_ = pickle.load(f)
            results.extend(batch_)


        print(f"Benchmark Inference Tests completed with {n} samples")

        with open(run_filepath, "wb") as f:
            pickle.dump(results, f)
            print(f"Successfully dumped test results to cache: \n\t{run_filepath}")

        if clear_cache and run_filepath.exists():
            print(f"Removing cache file: {run_filepath}")
            for file in batch_files:
                file.unlink()
                print(f"Removed batch file: {file}")

    return results
