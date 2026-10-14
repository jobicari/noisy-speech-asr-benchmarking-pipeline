from pathlib import Path
import platform
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv(override=False)


def _env_list(name, default):
    """Helper function to retrieve env as list, split on commas and remove any unintenional spaces"""
    raw = os.getenv(name, default)
    return [m.strip() for m in raw.split(",") if m.strip()]



@dataclass
class Config:
    SEED: int = field(default_factory=lambda: int(os.getenv("SEED", 42)))

    # Run type
    TEST_TYPE: str = field(default_factory=lambda: os.getenv("TEST_TYPE", ""))
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
    INFERENCE_BACKEND: str = field(
        default_factory=lambda: os.getenv("INFERENCE_BACKEND", "hf").lower().strip("")
    )

    REQUIRE_CUDA: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_CUDA", "True").lower() in ("true", "1", "yes")
    )

    MLX_MODELS: list = field(
        default_factory=lambda: _env_list(
            "MLX_MODELS", "mlx-community/whisper-small-mlx,mlx-community/whisper-large-v3-turbo"
        )
    )

    HF_MODELS: list = field(
        default_factory=lambda: _env_list(
            "HF_MODELS", "openai/whisper-small,openai/whisper-large-v3-turbo"
        )
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
    RUN_POSTFIX: str = field(
        default_factory=lambda: os.getenv("RUN_POSTFIX", "")
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

    N_PREP_WORKERS: int = field(
        default_factory=lambda: int(os.getenv("N_PREP_WORKERS", 1))
    )


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

        self.INFERENCE_BACKEND = self.INFERENCE_BACKEND.lower().strip()
        if self.INFERENCE_BACKEND not in ["mlx", "hf"]:
            raise ValueError(f"INFERENCE_BACKEND must be either 'mlx' or 'hf'. Got {self.INFERENCE_BACKEND}")
        if self.INFERENCE_BACKEND == "mlx" and not self.IS_APPLE_SILICON:
            raise ValueError(f"INFERENCE_BACKEND=mlx needs Apple Silicon. Use INFERENCE_BACKEND=hf on this machine.")

        self.TEST_TYPE = self.TEST_TYPE.strip()
        if self.TEST_TYPE not in ["smallDr", "fullDr", "fullRun"]:
            raise ValueError(f"TEST_TYPE must be either 'smallDr', 'fullDr' or 'fullRun'. Got {self.TEST_TYPE}")


    @property
    def MODELS(self):
        models = self.MLX_MODELS if self.INFERENCE_BACKEND == "mlx" else self.HF_MODELS
        if not models:
            raise ValueError("No models specified.")
        return models



