"""Tests that need no GPU, no network and no cloud credentials.

    python -m unittest discover -s tests -t . -v

Run from the repo root with the project's Python environment (numpy, scipy, librosa,
jiwer, pandas and google-cloud-storage installed). torch and transformers are never
imported: the model loader is tested against stand-ins.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Config() runs at import time in most modules and validates these
os.environ.setdefault("TEST_TYPE", "smallDr")
os.environ.setdefault("INFERENCE_BACKEND", "hf")
os.environ["USE_CLOUD_STORAGE"] = "True"   # no local results/ folders created on import
os.environ["SEED"] = "123"

for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
