import time
import jiwer
from asr_pipeline_benchmark.models import load_transcriber_engine
from asr_pipeline_benchmark.preprocessing import cleanup_pipeline

from asr_pipeline_benchmark.configuration import Config
config = Config()


def run_framework_speed_comparison(dataset, n_samples=10):
    # 1.load both engines
    force_mps = config.FORCE_MPS or True
    mlx_transcriber = load_transcriber_engine(IS_APPLE_SILICON=True)
    hf_transcriber = load_transcriber_engine(IS_APPLE_SILICON=False, FORCE_MPS=force_mps)

    # 2.Pre-load all samples to prevent biasing results from network latencies
    iterator = iter(dataset)
    samples = [next(iterator) for _ in range(n_samples + 1)]
    warmup_audio = samples[0]["audio"]["array"]
    test_samples = samples[1:]

    print(f"Warming up engines...")
    _ = mlx_transcriber(warmup_audio)
    _ = hf_transcriber(warmup_audio)

    # Pre-load ground truth text refs
    refs = []
    for s in test_samples:
        refs.append(cleanup_pipeline(s["text"]))

    # 3. Time engine A (MLX)
    print(f"Profiling Native MLX...")
    mlx_hyps = []
    mlx_start = time.perf_counter()
    for s in test_samples:
        mlx_hyps.append(cleanup_pipeline(mlx_transcriber(s["audio"]["array"])["text"]))
    mlx_time = time.perf_counter() - mlx_start

    # 4. Time engine 4 (PyTorch MPS)
    print(f"Profiling PyTorch {'mps' if force_mps else 'cpu'} pipeline...")
    hf_hyps =  []
    hf_start = time.perf_counter()
    for s in test_samples:
        hf_hyps.append(cleanup_pipeline(hf_transcriber(s["audio"]["array"])["text"]))
    hf_time = time.perf_counter() - hf_start

    # 5. Compute Metrics
    mlx_wer = jiwer.wer(refs, mlx_hyps)
    hf_wer = jiwer.wer(refs, hf_hyps)

    print(f"\nMLX Total Time   : {mlx_time:.2f}s | WER: {mlx_wer*100:.2f}%")
    print(f"PyTorch {'mps' if force_mps else 'cpu'} Time : {hf_time:.2f}s | WER: {hf_wer*100:.2f}%")
    print(f"MLX Speed Factor : {hf_time / mlx_time:.2f}x faster")

    return