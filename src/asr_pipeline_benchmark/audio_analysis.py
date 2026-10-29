import numpy as np
import librosa
from typing import Dict, Any
import soundfile as sf

def analyze_asr_audio_health(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> Dict[str, Any]:
    """
    Calculates audio metrics and flags structural issues that degrade ASR
    Assumes y is a 1D float32 numpy array
    Returns a dictionary of metrics and ASR flags
    """
    eps = 1e-10

    # ----------------------------------------
    # 1. Base energy and metrics
    # ----------------------------------------
    # Global RMS
    rms_linear_global = np.sqrt(np.mean(y**2))
    rms_dbfs_global = 20 * np.log10(np.maximum(rms_linear_global, eps))

    # Frame-based RMS
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_dbfs_frames = 20 * np.log10(np.maximum(rms_frames, eps))

    # Linear and DBFS Peaks
    abs_y = np.abs(y)
    peak_max_linear = np.max(abs_y)
    peak_max_dbfs = 20 * np.log10(np.maximum(peak_max_linear, eps))

    # ----------------------------------------
    # 2. Spectral Metrics
    # ----------------------------------------
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Spectral Flatness
    spect_flat = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)
    sf_mean = np.mean(spect_flat)
    sf_std = np.std(spect_flat)


    # ----------------------------------------
    # 3. ASR Specific Diagnostic Metrics
    # ----------------------------------------
    # DC Offset: Should be very close to 0.0
    dc_offset = np.mean(y)

    # Crest Factor: Peak to RMS Ratio in dB
    crest_factor_db = peak_max_dbfs - rms_dbfs_global
    #
    # Zero-Crossing Rate: High ZCR + Low RMS = Pure Noise
    zcr_frames = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_mean = np.mean(zcr_frames)

    # ----------------------------------------
    # 4. Automated ASR Health Flags
    # ----------------------------------------
    # Clip Detection: Values Tracking exactly at 1.0 or -1.0
    is_clipping = np.sum(abs_y >=.99) > (len(y) * 0.001)  # More than 0.1% of sampels

    # Too Quiet Detection: ASR models struggle below -40 dBFS RMS
    is_too_quiet = rms_dbfs_global < -40.0

    # Pure Noise/Silence Detection: High zero crossing but no real energy
    # is_pure_noise = (zcr_mean > 0.15) and (rms_dbfs_global < -30.0)

    zcr_min = np.min(zcr_frames)    # Vowels always drop ZCR low, noise dow not
    zcr_std = np.std(zcr_frames)    # Speech has high ZCR variance; static has low variance
    # If the minimum frame ZCR is high, it means the signal never became harmonic (no vowels)
    is_pure_noise = (zcr_min > 0.12) and (rms_dbfs_global < -25.0)

    round_factor = 4
    return {
        "metrics": {
            "rms_lin_global": round(float(rms_linear_global), round_factor),
            "rms_dbfs_global": round(float(rms_dbfs_global), round_factor),
            "rms_dbfs_min_frame": round(float(np.min(rms_dbfs_frames)), round_factor),
            "rms_dbfs_max_frame": round(float(np.max(rms_dbfs_frames)), round_factor),
            "linear_range": [round(float(np.min(y)), round_factor), round(float(np.max(y)), round_factor)],
            "peak_dbfs": round(float(peak_max_dbfs), round_factor),
            "dc_offset": round(float(dc_offset), round_factor),
            "crest_factor_db": round(float(crest_factor_db), round_factor),
            "min_zcr": round(float(zcr_min), round_factor),
            "std_zcr": round(float(zcr_std), round_factor),
            "mean_zcr": round(float(zcr_mean), round_factor),
            "mfccs_mean": [round(x, round_factor) for x in mfccs_mean.tolist()],
            "mfccs_std": [round(x, round_factor) for x in mfccs_std.tolist()],
            "sf_mean": sf_mean,
            "sf_std": sf_std
        },
        "asr_flags": {
            "action_required": bool(is_clipping or is_too_quiet or is_pure_noise or (abs(dc_offset) > .01)),
            "reasons": {
                "digital_clipping_detected": bool(is_clipping),
                "signal_too_faint": bool(is_too_quiet),
                "likely_static_or_hum": bool(is_pure_noise),
                "dc_offset_warning": bool(abs(dc_offset) > 0.01)
            }
        }
    }