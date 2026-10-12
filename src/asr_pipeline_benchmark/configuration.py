from pathlib import Path
import platform
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv(override=True)

@dataclass
class Config:
    SEED: int = field(default_factory=lambda: int(os.getenv("SEED", 42)))

    # Run type
    TEST_TYPE: str = field(default_factory=lambda: os.getenv("TEST_TYPE", "smallDR"))
    NUMBER_OF_SAMPLES: int = field(default_factory=lambda: int(os.getenv("NUMBER_OF_SAMPLES", 10)))

    # ----------------------------------------------------
    # Credentials
    # ----------------------------------------------------
    USE_CLOUD_STORAGE: bool = field(
        default_factory=lambda: os.getenv("USE_CLOUD_STORAGE", "False").lower() in ("true", "1", "yes")
    )
    GCS_BUCKET_NAME: str = field(
        default_factory=lambda: os.getenv("GCS_BUCKET_NAME")
    )


    # ----------------------------------------------------
    # Datasets
    # ----------------------------------------------------
    SPEECH_DATASET: str = field(
        default_factory=lambda: os.getenv("SPEECH_DATASET")
    )
    IR_DATASET: str = field(
        default_factory=lambda: os.getenv("IR_DATASET")
    )


    # ----------------------------------------------------
    # Dynamic Path & Key Resolution
    # ----------------------------------------------------
    REL_TRANSCRIBE_CACHE_DIR: str = "results/transcribe_cache"
    REL_CHECKPOINT_DIR: str = "results/transcribe_cache/checkpoint"

    # 3. Dynamic Fields (init=False prevents dataclass from expecting them as init args)
    TRANSCRIBE_CACHE_DIR: str | Path = field(init=False)
    CHECKPOINT_DIR: str | Path = field(init=False)
    PROJECT_ROOT: Path = field(init=False)

    def __post_init__(self):
        # Anchor project  root dynamically relative to config file location
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

        if self.USE_CLOUD_STORAGE:
            # GCS Object Keys MUST be clean, relative POSIX strings (no leading / or OS drive letters)
            self.TRANSCRIBE_CACHE_DIR = self.REL_TRANSCRIBE_CACHE_DIR
            self.CHECKPOINT_DIR = self.REL_CHECKPOINT_DIR
        else:
            # Local filesystem uses absolute Path objects anchored to PROJECT_ROOT
            self.TRANSCRIBE_CACHE_DIR = self.PROJECT_ROOT / self.REL_TRANSCRIBE_CACHE_DIR
            self.CHECKPOINT_DIR = self.PROJECT_ROOT / self.REL_CHECKPOINT_DIR

            Path(self.TRANSCRIBE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            Path(self.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


    # ----------------------------------------------------
    # OPERATION FLAGS
    # ----------------------------------------------------
    RUN_NOISE_TESTS: bool = field(
        default_factory=lambda: os.getenv("RUN_NOISE_TESTS", "False").lower() in ("true", "1", "yes")
    )
    LAUNCH_GRADIO: bool = field(
        default_factory=lambda: os.getenv("LAUNCH_GRADIO", "False").lower() in ("true", "1", "yes")
    )

    # Flag if running on Apple Silicon (M1) to load MLX-WHISPER for MLX models.
    IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

    # == MODEL FLAGS ==
    # MODEL_SMALL determines which size of transcriber engines is loaded.
    # Based on the model names below
    MODEL_SMALL: bool = field(
        default_factory=lambda: os.getenv("MODEL_SMALL", "True").lower() in ("true", "1", "yes")
    )
    MLX_SMALL_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("MLX_SMALL_MODEL_NAME", "mlx-community/whisper-small-mlx")
    )
    MLX_LARGE_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("MLX_LARGE_MODEL_NAME", "mlx-community/whisper-large-v3-turbo")
    )
    HF_TRANSFORMER_SMALL_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("HF_TRANSFORMER_SMALL_MODEL_NAME", "openai/whisper-small")
    )
    HF_TRANSFORMER_LARGE_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("HF_TRANSFORMER_LARGE_MODEL_NAME", "openai/whisper-large-v3-turbo" )
    )


    # ----------------------------------------------------
    # BENCHMARK FLAGS
    # ----------------------------------------------------
    # inference runs will verify first if a previous run has been cached as pickle file in TRANSCRIBE_CACHE_DIR
    # and load it first if available. Otherwise will run inference.
    # Force train runs inference if previous run is present and overwrites the file.
    # runs are labeled as such: <type>__<modelname>__<numberOfSamples>__<postfixInfo>.pkl
    # if any of those variables do not match, the inference run is considered different and will run.
    FORCE_RETRAIN: bool = field(
        default_factory=lambda: os.getenv("FORCE_RETRAIN", "False").lower() in ("true", "1", "yes")
    )
    RUN_BENCHMARK_INFERENCE: bool = field(
        default_factory=lambda: os.getenv("RUN_BENCHMARK_INFERENCE", "True").lower() in ("true", "1", "yes")
    )
    # For future implementation. Not used at the moment.
    IS_MOCK: bool = field(
        default_factory=lambda: os.getenv("IS_MOCK", "True").lower() in ("true", "1", "yes")
    )
    BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", 1))
    )
    CHECKPOINT_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHECKPOINT_SIZE", 50))
    )

    # ----------------------------------------------------
    # SPEED COMPARISON FLAGS
    # ----------------------------------------------------
    RUN_FRAMEWORK_SPEED_COMPARISON_TEST: bool = field(
        default_factory=lambda: os.getenv("RUN_FRAMEWORK_SPEED_COMPARISON_TEST", "False").lower() in (
            "true", "1", "yes"
        )
    )

    # To run frameworkd comparison between 'mlx' and 'mps', need to force it.
    # Otherwise, will run on 'cpu'
    FORCE_MPS: bool = field(
        default_factory=lambda: os.getenv("FORCE_MPS", "False").lower() in ("true", "1", "yes")
    )





