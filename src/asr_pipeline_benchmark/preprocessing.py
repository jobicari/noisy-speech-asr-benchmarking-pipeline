import subprocess
import tempfile
from pathlib import Path
import soundfile as sf

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
import jiwer

from asr_pipeline_benchmark.data import get_mit_ir_dataset
from asr_pipeline_benchmark.configuration import Config
config = Config()

########################
### HELPER FUNCTIONS ###
########################

def cleanup_pipeline(text: str) -> str:
    """Normalize ASR transcriptions by lowercasing, removing punctuation, and stripping whitespace."""
    pipeline = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
    ])
    return pipeline(text)


def extract_ir_from_mit_dataset(dataset=None, ir_idx=None, target_sr=16_000):
    """
    Retrieve the IR audio array from the MIT Impulse Response survey dataset based on its id.
    If dataset is not provided, the default MIT IR survey is lazy-loaded and cached.
    """
    if dataset is None:
        dataset = get_mit_ir_dataset()
    sample = dataset.filter(lambda sample: sample["id"]==ir_idx)
    decoder = sample["audio"][0]
    audio_tensor = decoder.get_all_samples()
    y_native = audio_tensor.data.numpy()
    sr_native = sample["audio"][0].metadata.sample_rate

    y = librosa.resample(y_native, orig_sr=sr_native, target_sr=target_sr)

    return y, target_sr


def extract_irs_from_mit_dataset_by_location(dataset, location, target_sr=16_000):
    """
    Retreive the IR audio array fromthe MIT Impulse Respons survey dataset based on its id
    """
    examples = dataset.filter(lambda sample: sample["location"]==location)

    results = []

    for example in examples:
        idx = example["id"]
        detail = example["detail"]
        decoder = example["audio"]
        audio_tensor = decoder.get_all_samples()
        y_native = audio_tensor.data.numpy()
        sr_native = example["audio"].metadata.sample_rate
        duration = example["audio"].metadata.duration_seconds

        y = librosa.resample(y_native, orig_sr=sr_native, target_sr=target_sr)

        results.append(
            {'id': idx,
             'location': location,
             'detail': detail,
             'duration': duration,
             'audio': y,
             'sr': target_sr
             })

    return results


############################
### CORRUPTION FUNCTIONS ###
############################


def add_white_noise(signal, target_snr_db):
    """
    Inject additive white Gaussian noise scaled to a precise target SNR (dB)
    """
    # Calculate RMS of clean signal
    rms_signal = np.sqrt(np.mean(np.square(signal)))

    # Calculate the required RMS of noise based on target SNR in dB
    rms_noise = rms_signal / (10**(target_snr_db/20))

    # Generate noise and scale it
    noise_raw = np.random.randn(len(signal))
    rms_noise_raw = np.sqrt(np.mean(np.square(noise_raw)))

    # Scale the noise to match the target noise RMS
    noise_scaled = noise_raw * (rms_noise / rms_noise_raw)

    signal_with_noise = signal + noise_scaled

    if np.max(np.abs(signal_with_noise))>1.0:
        print("Warning: Mixed audio exceeds 0 dBFS. Normalizing to prevent clipping.")
        signal_with_noise = signal_with_noise / np.max(np.abs(signal_with_noise))

    return signal_with_noise


def add_synthetic_reverberation(signal, sr=16_000, rt60=0.5, mix=0.4):
    """
    Synthesizes an exponential decay Impulse Response (IR) to simulate room acoustics,
    then convolve it with the clean speech signal

    rt6-: Time (seconds) for the reverb tail to drop by 60dB.
    mix: dry/wet mix balance between direct signal and reverberation
    """
    ir_len = int(rt60 * sr)
    t = np.linspace(0, rt60, ir_len)

    # Generate white noise and apply clean mathematical exponential devay envelope
    # -6.91 guarantees the amplitude drops exactly 60 dB over the course of rt60
    decay_envelope = np.exp(-6.91 * t / rt60)
    ir = np.random.randn(ir_len) * decay_envelope

    # Normalize the IR energy fader to prevent massive volume scaling jumps
    # In signal processing, we use L2 normalization to keep the total amount of enery = 1
    ir = ir / np.sqrt(np.sum(ir**2))

    # Convolve the signal with our custom synthetic room IR
    reverb_signal = np.convolve(signal, ir, mode='full')[:len(signal)]

    # Apply standard Wet/Dry mixing balance
    mixed_signal = ( 1-mix ) * signal + mix * reverb_signal

    # Safety rail guard to prevent hard clipping
    if np.max(np.abs(mixed_signal)) > 1.0:
        mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

    return mixed_signal


def add_real_reverberation(signal, ir, sr=16_000, mix=0.4):
    """
    Convolves a clean audio signal with a recorder physical Impulse Response
    """

    # Force mono on both signal and IR
    # if len(signal.shape)> 1:
    #     signal = np.mean(signal, axis=1)
    # if len(ir.shape) > 1:
    #     ir = np.mean(ir, axis=0)

    # Flatten ir array if necessary
    if ir.shape[0]==1:
        ir = ir.flatten()


    # Normalize the IR energy so it doesn't artificially boost the overall volume
    rms = np.sqrt(np.mean(ir**2))
    target_rms = 0.05
    ir_scaled = ir * (target_rms / rms)
    ir_scaled = np.clip(ir_scaled, -1.0, 1.0)

    # Convolve the signal with the real room IR
    reverb_signal = np.convolve(signal, ir_scaled, mode='full')[:len(signal)]

    # Apply the wer/dry balance
    mixed_signal = ( 1 - mix ) * signal + mix * reverb_signal


    # Safety prevention of clipping
    if np.max(np.abs(mixed_signal)) > 1.0:
        mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

    return mixed_signal



def add_smart_audio_dropouts(signal, sr=16_000, num_dropouts=8, dropout_duration_ms=100, threshold_below_global_db=15):
    """
    Simulates network packet loss or a bad connection by zeroing out
    chunks of audio at a random interval timestamps
    deliberately targeting high energy sections (above threshold_below_global_db)
    and avoiding silence parts (below threshold_below_global_db)
    """
    degraded_signal = signal.copy()
    dropout_len = int((dropout_duration_ms / 1000.0) * sr)

    # Ensure dropout zone is bounded within track length
    if len(signal) <= dropout_len:
        return degraded_signal

    possible_starts = len(signal) - dropout_len

    dropouts_applied = 0
    attempts = 0
    max_attempts = 50
    eps = 1e-10
    rms_linear_global = np.sqrt(np.mean(signal**2))
    rms_dbfs_global = 20 * np.log10(np.maximum(rms_linear_global, eps))

    adaptive_silence_threshold = rms_dbfs_global - threshold_below_global_db

    while dropouts_applied < num_dropouts  and attempts < max_attempts:
        # Pick a random start index
        start = np.random.randint(0, possible_starts)

        # Extract the targeted chunk of audio
        target = signal[start: start + dropout_len]

        # Claculate RMS energy of this specific chunk and convert to dBFS
        target_rms_linear = np.sqrt(np.mean(target**2))
        target_rms_dbfs = 20 * np.log10(np.maximum(target_rms_linear, eps))

        if target_rms_dbfs > adaptive_silence_threshold:
            # Accepted chunk as speech
            degraded_signal[start : start + dropout_len] = 0.0
            dropouts_applied+=1

        attempts+=1

    return degraded_signal


def packet_loss(audio, sr=16_000, packet_ms=20, loss_rate=0.05,
                mode="zero", seed=None):
    """
    Simulate packet loss.

    mode:
        "zero"   -> lost packets become silence
        "repeat" -> repeat previous packet
        "noise"  -> replace with low-level noise
    """
    if seed is not None:
        rng = np.random.seed(seed)
    else:
        rng = np.random.default_rng(config.SEED)

    packet_size = int(sr * packet_ms / 1000)
    output = audio.copy()

    for start in range(0, len(audio), packet_size):
        end = min(start + packet_size, len(audio))

        if rng.random() < loss_rate:
            if mode == "zero":
                output[start:end] = 0

            elif mode == "repeat" and start >= packet_size:
                previous = output[start - packet_size:start]
                n = end - start
                output[start:end] = previous[:n]

            elif mode == "noise":
                noise = rng.normal(0, 0.001, end - start)
                output[start:end] = noise

    return output




def band_limit(audio, sr=16_000, lowcut=None, highcut=None, order=6):
    """
    Apply low-pass, high-pass, or band-pass filtering.
    """

    nyquist = sr / 2

    if lowcut is not None and highcut is not None:
        sos = butter(
            order,
            [lowcut / nyquist, highcut / nyquist],
            btype="band",
            output="sos"
        )

    elif lowcut is not None:
        sos = butter(
            order,
            lowcut / nyquist,
            btype="highpass",
            output="sos"
        )

    elif highcut is not None:
        sos = butter(
            order,
            highcut / nyquist,
            btype="lowpass",
            output="sos"
        )

    else:
        return audio.copy()

    return sosfiltfilt(sos, audio)


def hard_clip_ratio(audio, threshold=0.8):
    """
    Normalize to peak amplitude and then hard-clip
    :param audio:
    :param threshold:
    :return:
    """

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio.copy()

    return np.clip(audio, -(threshold*peak), (threshold*peak))

def hard_clip_ratio_norm(audio, threshold=0.8):
    """
    Normalize to peak amplitude and then hard-clip
    :param audio:
    :param threshold:
    :return:
    """

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio.copy()
    normalize = audio / peak
    return np.clip(normalize, -threshold, threshold)



# codec -> (container extension, ffmpeg muxer, minimum supported bitrate in kbps)
_CODEC_SPECS = {
    "libopus":    (".opus", "opus", 6),
    "libvorbis":  (".ogg",  "ogg",  45),   # Vorbis has a hard ~45 kbps floor
    "libmp3lame": (".mp3",  "mp3",  8),
}


def _bitrate_to_kbps(bitrate: str) -> int:
    """Parse '8k' / '32k' / '64000' style bitrate strings into kbps."""
    b = bitrate.strip().lower()
    if b.endswith("k"):
        return int(float(b[:-1]))
    return int(float(b) / 1000)


def codec_degrade(audio, sr=16_000, bitrate="16k", codec="libopus"):
    """Round-trip `audio` through an ffmpeg encode/decode for the given codec.

    Returns a mono float32 numpy array at the original `sr`.
    """
    if codec not in _CODEC_SPECS:
        raise ValueError(f"Unsupported codec {codec!r}. "
                         f"Choose from {sorted(_CODEC_SPECS)}")

    ext, fmt, min_kbps = _CODEC_SPECS[codec]
    kbps = _bitrate_to_kbps(bitrate)

    # Fail fast with a clear message instead of letting ffmpeg die opaquely.
    if kbps < min_kbps:
        raise ValueError(
            f"{codec} cannot encode at {bitrate} "
            f"(minimum is {min_kbps}k). Use a higher bitrate for this codec."
        )

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:                       # force mono for a fair comparison
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0)        # avoid clipping artifacts

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_path = tmp / "input.wav"
        output_path = tmp / f"output{ext}"

        sf.write(input_path, audio, sr)

        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-c:a", codec,
            "-b:a", f"{kbps}k",
            "-f", fmt,                       # explicit container for the codec
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Surface the real ffmpeg diagnostic rather than a bare exit code.
            raise RuntimeError(
                f"ffmpeg failed for codec={codec!r}, bitrate={bitrate!r} "
                f"(exit {result.returncode}):\n{result.stderr.strip()}"
            )

        degraded, out_sr = sf.read(output_path, dtype="float32")

    if degraded.ndim > 1:
        degraded = degraded.mean(axis=1)

    # Opus/Vorbis/MP3 decode at their native rates (Opus -> 48 kHz);
    # resample back so every variant aligns with the original signal.
    if out_sr != sr:
        n_out = int(round(len(degraded) * sr / out_sr))
        degraded = np.interp(
            np.linspace(0, 1, n_out),
            np.linspace(0, 1, len(degraded)),
            degraded,
        ).astype(np.float32)

    return degraded