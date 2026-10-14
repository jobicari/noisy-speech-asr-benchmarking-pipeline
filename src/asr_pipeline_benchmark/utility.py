import json
from datetime import datetime, timezone
import time
import hashlib
import zlib
import numpy as np
import pandas as pd
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



def _hash_model_list(model_list):
    return hashlib.sha1(str(model_list).encode()).hexdigest()[:8]


def generate_run_label(test_type, model_backend, models,  n_samples, postfix=""):
    models_sorted = sorted(models)
    model_hash = _hash_model_list(models_sorted)

    if len(postfix) > 0:
        return f"{test_type}__{model_backend}__{model_hash}__{n_samples}samples__{postfix}"
    else:
        return f"{test_type}__{model_backend}__{model_hash}__{n_samples}samples"


def checkpoint_iterator(iterable, checkpoint_size=50):
    """Yields small lists of samples (micro-batches) from a streaming iterator."""
    cp_batch = []
    for item in iterable:
        cp_batch.append(item)
        if len(cp_batch) == checkpoint_size:
            yield cp_batch
            cp_batch = []
    if cp_batch:  # Yield any remaining tail items
        yield cp_batch


def cleanup_results_col_names(results):
    """
    Take the list of results which contains nested dictionnaries, flattens them, and,
    renames columns with more friendly names.
    :param results: List
    :return: dataframe
    """

    # Flattens nestsed dictionnaries
    df = pd.json_normalize(results)

    # Splits colnames
    cols = [col.split(".") for col in df.columns]

    # Columns mid-labels to exclude
    exclude = {"sample_analysis", "reasons"}

    cleaned_cols = [
        ".".join(
            item.replace("metrics", "mtr")
            .replace("asr_flags", "flg")
            .replace("action_required", "act_req")
            .replace("digital_clipping_detected", "clip_det")
            .replace("signal_too_faint", "sig_low")
            .replace("likely_static_or_hum", "stat_hum")
            .replace("dc_offset_warning", "dc_warn")
            for item in sublist
            if item not in exclude
        )
        for sublist in cols
    ]

    df.columns = cleaned_cols
    return df


def expand_mfcc_results(df):
    """
    Takes the dataframe cleanep up by cleanup_results_col_names and expands the mfcc columns.
    :param df:
    :return: expanded df
    """
    df_copy = df.copy()

    columns = df_copy.columns.tolist()
    cols_to_expand = ["mtr.mfccs_mean", "mtr.mfccs_std"]
    mean_col_idx = df_copy.columns.get_loc(cols_to_expand[0])
    std_col_idx = df_copy.columns.get_loc(cols_to_expand[1])

    expanded_cols = pd.concat([
        pd.DataFrame(df_copy[col].tolist(), index=df.index).rename(
            columns=lambda x, c=col: f"{c.split('_')[0].rstrip('s')}{x+1}_{c.split('_')[1]}"
        )
        for col in cols_to_expand
    ], axis=1)

    new_col_order = (
        columns[:mean_col_idx] +
        expanded_cols.columns.tolist() +
        columns[std_col_idx+1:]
    )

    return pd.concat([df_copy.drop(columns=cols_to_expand), expanded_cols], axis=1)[new_col_order]
    # return expanded_cols



def print_timing(record):
    utc_now = datetime.now(timezone.utc)
    print(f"[TIMING] {json.dumps({'ts': utc_now.isoformat(), **record})}", flush=True)



def _sample_seed(sample_id):
    """Generate a deterministic seed for a given sample ID"""
    return (config.SEED + zlib.crc32(str(sample_id).encode())) % 2**32



def _seed(sample_id, corr_type):
    """Generate a deterministic seed for a given sample ID and correlation type"""
    key = f"{sample_id}_{corr_type}".encode()
    return (config.SEED + zlib.crc32(key)) % 2**32