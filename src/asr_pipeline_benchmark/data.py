"""Dataset loading and caching for the ASR benchmark pipeline."""

_mit_ir_dataset = None

def get_mit_ir_dataset():
    """Lazy-load and cache the MIT impulse response survey dataset."""
    global _mit_ir_dataset
    if _mit_ir_dataset is None:
        from datasets import load_dataset
        _mit_ir_dataset = load_dataset("benjamin-paine/mit-impulse-response-survey", split="train")
    return _mit_ir_dataset
