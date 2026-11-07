FROM python:3.12-slim

# System deps:
# - ffmpeg: preprocessing.py shells out to the ffmpeg binary directly (codec_degrade)
# - libsndfile1: required by the soundfile package (audio_analysis.py, preprocessing.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# One install step for everything. torch, torchaudio and torchcodec are pinned in
# requirements-docker.txt and come from the cu126 index; constraints-docker.txt pins
# every other package, so a rebuild runs the same code.
# cu126 wheels bundle their own CUDA runtime, so no CUDA base image is needed —
# only the host's NVIDIA driver + nvidia-container-toolkit at `docker run` time.
# torch must stay >= 2.6: transformers assumes torch.accelerator, and an older build
# crashes its import with "no attribute 'accelerator'".
# torchcodec must be the CUDA build that matches torch: datasets' audio decoding
# backend requires it, it's not optional despite not being imported in src/.

COPY requirements-docker.txt constraints-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt -c constraints-docker.txt

COPY src/ /app/src/
COPY app.py /app/app.py

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# The 'app' user is a system account (useradd -r) with no home directory, so
# $HOME defaults to /home/app, which doesn't exist and can't be created at
# runtime. Point HOME/HF_HOME at a writable path under /app instead, so
# huggingface_hub's model/dataset cache has somewhere to write.
ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

RUN chown -R app:app /app
USER app

ENTRYPOINT ["python", "app.py"]
