FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CosyVoice isn't pip-installable — it's vendored as source, per upstream's
# own install instructions (github.com/FunAudioLLM/CosyVoice). --recursive
# pulls the third_party/Matcha-TTS git submodule it depends on at import time.
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

COPY requirements.txt .
# Trimmed from upstream's own requirements.txt: dropped deepspeed, grpcio(-tools),
# gradio, fastapi(-cli), and the three tensorrt-cu12* packages. Checked against
# upstream's source: none are imported by the inference path (cli/cosyvoice.py,
# cli/model.py, cli/frontend.py, utils/*) — they're only used by training
# scripts, the gRPC/FastAPI server, and TensorRT acceleration (load_trt=True,
# which we never set). The import check below fails the build if that turns
# out to be wrong for any module the model actually pulls in.
# openai-whisper (and possibly other sdist-only packages) build from source,
# and their setup.py imports pkg_resources — which the newest setuptools no
# longer ships, and pip's isolated build env always pulls the newest one
# (this failed the first CI build). Pin a setuptools that still has it plus
# the build tools sdists need (pyworld wants cython+numpy), then build
# without isolation so those are the ones used.
RUN pip install "setuptools<81" wheel cython numpy==1.26.4
RUN pip install --no-build-isolation --default-timeout=200 -r requirements.txt

RUN python -c "import sys; sys.path.extend(['/app/CosyVoice', '/app/CosyVoice/third_party/Matcha-TTS']); import runpod; from cosyvoice.cli.cosyvoice import AutoModel; print('import check ok')"

# Badini fine-tuned checkpoint (~6.75GB: CosyVoice3 weights + the
# CosyVoice-BlankEN LLM backbone its own cosyvoice3.yaml requires).
ARG MODEL_ID=computeram/cosyvoice3-badini-tts
ENV MODEL_ID=${MODEL_ID}
RUN python - << 'PYEOF'
import os
from huggingface_hub import snapshot_download
model_id = os.environ.get("MODEL_ID")
path = snapshot_download(model_id, local_dir="/app/model")
print(f"Downloaded {model_id} to {path}")
PYEOF

# Bundled reference voice: a real Badini recording + its exact transcript.
# CosyVoice3 is a zero-shot cloning model, so every generation needs a
# reference clip; callers can override it with ref_audio/ref_text.
COPY prompt-hayfa.wav prompt-hayfa.txt /app/

COPY runpod_handler.py .

ENV PYTHONPATH=/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS
ENV MODEL_DIR=/app/model

CMD ["python", "-u", "runpod_handler.py"]
