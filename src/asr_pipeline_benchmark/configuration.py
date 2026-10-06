from pathlib import Path
import platform
from dataclasses import dataclass


@dataclass
class Config:
    # Datasets
    SPEECH_DATASET = "openslr/librispeech_asr"
    IR_DATASET = "benjamin-paine/mit-impulse-response-survey"


    # == PATHS ==
    SEED = 42

    NOTEBOOK_DIR = Path.cwd()
    TRANSCRIBE_CACHE_PATH = NOTEBOOK_DIR / "results" / "transcribe_cache"
    CHECKPOINT_DIR = TRANSCRIBE_CACHE_PATH / "checkpoint"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    RUN_NOISE_TESTS = False
    LAUNCH_GRADIO = True

    # Flag if running on Apple Silicon (M1) to load MLX-WHISPER for MLX models.
    IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

    # == MODEL FLAGS ==
    # MODEL_SMALL determines which size of transcriber engines is loaded.
    # Based on the model names below
    MODEL_SMALL = True  # Set False to run large models
    MLX_SMALL_MODEL_NAME = "mlx-community/whisper-small-mlx"
    MLX_LARGE_MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
    HF_TRANSFORMER_SMALL_MODEL_NAME = "openai/whisper-small"
    HF_TRANSFORMER_LARGE_MODEL_NAME = "openai/whisper-large-v3-turbo"


    # == Benchmark Flags ==
    # inference runs will verify first if a previous run has been cached as pickle file in TRANSCRIBE_CACHE_PATH
    # and load it first if available. Otherwise will run inference.
    # Force train runs inference if previous run is present and overwrites the file.
    # runs are labeled as such: <type>__<modelname>__<numberOfSamples>__<postfixInfo>.pkl
    # if any of those variables do not match, the inference run is considered different and will run.
    FORCE_RETRAIN = True
    RUN_BENCHMARK_INFERENCE = True
    IS_MOCK = True  # For future implementation. Not used at the moment.
    BATCH_SIZE = 50

    # == Speed Comparison Flags ==
    RUN_FRAMEWORK_SPEED_COMPARISON_TEST = False
    # To run frameworkd comparison between 'mlx' and 'mps', need to force it.
    # Otherwise, will run on 'cpu'
    FORCE_MPS = True

