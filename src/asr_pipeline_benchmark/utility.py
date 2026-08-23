import numpy as np
from asr_pipeline_benchmark.configuration import Config
config = Config()

def generate_sine_tone(frequency=440.0, duration=2.0, sr=16_000, amplitude=1.0):
    """
    Generate a pure sine tone array perfectly compatible with librosa for testing.
    """
    # Create the time axis array
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Generate the sine wave and scale by amplitude (1.0 is full scale / 0 dBFS peak)
    y = amplitude * np.sin(2 * np.pi * frequency * t)

    # Ensure it uses float32
    return y.astype(np.float32), sr



def generate_run_label(test_type, model_backend, n_samples, MODEL_SMALL=True, postfix=""):
    if model_backend =="mlx":
        if MODEL_SMALL:
            model_name = config.MLX_SMALL_MODEL_NAME.replace("/", "_")
        else:
            model_name = config.MLX_LARGE_MODEL_NAME.replace("/", "_")
    else:
        if MODEL_SMALL:
            model_name = config.HF_TRANSFORMER_SMALL_MODEL_NAME.replace("/", "_")
        else:
            model_name = config.HF_TRANSFORMER_LARGE_MODEL_NAME.replace("/", "_")

    if len(postfix) > 0:
        return f"{test_type}__{model_name}__{n_samples}samples__{postfix}"
    else:
        return f"{test_type}__{model_name}__{n_samples}samples"


def batch_iterator(iterable, batch_size=50):
    """Yields small lists of samples (micro-batches) from a streaming iterator."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield any remaining tail items
        yield batch