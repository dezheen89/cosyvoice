"""
RunPod Serverless handler for CosyVoice3 (Badini Kurdish, zero-shot TTS).

Model: computeram/cosyvoice3-badini-tts (fine-tune of FunAudioLLM/CosyVoice3
on Badini Kurdish speech data). Zero-shot voice cloning — every generation
needs a reference audio clip + its exact transcript. This checkpoint ships
no reference audio of its own, so a bundled default (prompt-hayfa.wav/.txt,
a real Badini recording reused from Dezheen89/omnivoice-badini) is used
unless the caller supplies their own via ref_audio/ref_text.

Input:
    {"text": "...", "ref_audio": "<base64 wav>"?, "ref_text": "..."?, "speed": 1.0?}
Output:
    {"ok": true, "audio_base64": "...", "sample_rate": N, "format": "wav"}
"""

import base64
import io
import os
import sys
import threading
import time
import traceback

import runpod
import torchaudio

sys.path.append("/app/CosyVoice")
sys.path.append("/app/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
DEFAULT_REF_WAV = "/app/prompt-hayfa.wav"
DEFAULT_REF_TXT = "/app/prompt-hayfa.txt"
# Matches the model card's own verified usage example exactly.
DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"

with open(DEFAULT_REF_TXT, encoding="utf-8") as f:
    DEFAULT_REF_TEXT = f.read().strip()

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print(f"[LOAD] Loading CosyVoice3 from {MODEL_DIR}...", flush=True)
                _model = AutoModel(model_dir=MODEL_DIR)
                print("[LOAD] Model ready.", flush=True)
    return _model


def handler(job):
    try:
        input_data = job.get("input", {})
        text = input_data.get("text", "")
        if not text:
            return {"ok": False, "error": "Missing 'text' in input."}

        ref_wav = DEFAULT_REF_WAV
        ref_text = DEFAULT_REF_TEXT

        ref_audio_b64 = input_data.get("ref_audio")
        if ref_audio_b64:
            ref_text_in = input_data.get("ref_text")
            if not ref_text_in:
                return {"ok": False, "error": "ref_text is required with ref_audio."}
            tmp_path = f"/tmp/ref_{job.get('id', 'x')}.wav"
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(ref_audio_b64))
            ref_wav = tmp_path
            ref_text = ref_text_in

        speed = float(input_data.get("speed", 1.0))
        speed = max(0.5, min(2.0, speed))

        cosy = get_model()
        prompt = DEFAULT_INSTRUCT + ref_text

        t0 = time.perf_counter()
        result = None
        for out in cosy.inference_zero_shot(text, prompt, ref_wav, stream=False, speed=speed):
            result = out
            break
        if result is None:
            return {"ok": False, "error": "Model produced no output."}
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        buf = io.BytesIO()
        torchaudio.save(buf, result["tts_speech"], cosy.sample_rate, format="wav")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "ok": True,
            "audio_base64": audio_b64,
            "sample_rate": cosy.sample_rate,
            "format": "wav",
            "inference_ms": inference_ms,
        }
    except Exception:
        print(traceback.format_exc(), flush=True)
        return {"ok": False, "error": "Internal server error"}


runpod.serverless.start({"handler": handler})
