"""Dataset loading and caching for the ASR benchmark pipeline."""
from asr_pipeline_benchmark.configuration import Config
config = Config()

_mit_ir_dataset = None
_speech_dataset = None

def get_speech_dataset():
    """Lazy-load and cache the speech dataset (in streaming mode)."""
    global _speech_dataset
    if _speech_dataset is None:
        from datasets import load_dataset
        _speech_dataset = load_dataset(config.SPEECH_DATASET, "clean", split="test", streaming=True)
    return _speech_dataset


def get_mit_ir_dataset():
    """Lazy-load and cache the MIT impulse response survey dataset."""
    global _mit_ir_dataset
    if _mit_ir_dataset is None:
        from datasets import load_dataset
        _mit_ir_dataset = load_dataset(config.IR_DATASET, split="train")
    return _mit_ir_dataset
