"""
RunPod Serverless handler for CosyVoice3 (Badini Kurdish, zero-shot TTS).

Model: computeram/cosyvoice3-badini-tts (fine-tune of FunAudioLLM/CosyVoice3
on Badini Kurdish speech data). Zero-shot voice cloning — every generation
needs a reference audio clip + its exact transcript. This checkpoint ships
no reference audio of its own, so a bundled default (prompt-hayfa.wav/.txt,
a real Badini recording reused from Dezheen89/omnivoice-badini) is used
unless the caller supplies their own via ref_audio/ref_text, or picks one
of the bundled emotion references below by name.

The emotion references are real labeled clips (same Hayfa speaker) from a
Badini emotional-speech dataset — CosyVoice3 has no explicit emotion
conditioning, so "emotion" here just selects which real recording of this
speaker actually saying something in that emotional register is used as the
zero-shot clone target, which is the only way this model expresses it.

Input:
    {"text": "...", "emotion": "natural"|"angry"|"sad"|"happy"|"fear"|"surprise"?,
     "ref_audio": "<base64 wav>"?, "ref_text": "..."?, "speed": 1.0?}
Output:
    {"ok": true, "audio_base64": "...", "sample_rate": N, "format": "wav"}
"""

import base64
import io
import os
import re
import subprocess
import sys
import threading
import time
import traceback

import runpod
import torch
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

# Bundled per-emotion reference clips (see module docstring). Each maps to a
# (wav_path, text) pair, loaded once at startup; "natural" reuses the
# original default reference rather than a separate file.
EMOTION_FILES = {
    "angry": "/app/prompt-angry",
    "sad": "/app/prompt-sad",
    "happy": "/app/prompt-happy",
    "fear": "/app/prompt-fear",
    "surprise": "/app/prompt-surprise",
}
EMOTION_REFS = {"natural": (DEFAULT_REF_WAV, DEFAULT_REF_TEXT)}
for _emotion, _base in EMOTION_FILES.items():
    with open(f"{_base}.txt", encoding="utf-8") as _f:
        EMOTION_REFS[_emotion] = (f"{_base}.wav", _f.read().strip())

_model = None
_lock = threading.Lock()

# CosyVoice3's own internal splitter (frontend.text_normalize) only breaks on
# ASCII '.', '?', '!', ';', ':' — it doesn't recognize Arabic-script sentence
# punctuation ('،', '؟') that real Badini text uses heavily, and this dialect's
# text often runs long stretches with sparse punctuation at all. Left alone, a
# long paste can arrive as one giant unsplit chunk and overflow the model's
# generation length. So we pre-chunk by word count here (same soft/hard-cap
# approach already used for paragraph splitting elsewhere in this project)
# before ever handing text to the model, splitting at whichever kind of
# sentence-end punctuation appears, falling back to a hard word-count cut for
# a single very long run with none at all.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
CHUNK_SOFT_WORDS = 60
CHUNK_HARD_WORDS = 90


def chunk_text(text, soft_words=CHUNK_SOFT_WORDS, hard_words=CHUNK_HARD_WORDS):
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text]

    chunks = []
    cur, cur_words = [], 0
    for sent in sentences:
        words = sent.split()
        if len(words) > hard_words:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_words = [], 0
            for i in range(0, len(words), hard_words):
                chunks.append(" ".join(words[i:i + hard_words]))
            continue
        if cur_words and (cur_words + len(words) > hard_words or cur_words >= soft_words):
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.extend(words)
        cur_words += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


_model_fp16 = None


def get_model(fp16=False):
    global _model, _model_fp16
    with _lock:
        if _model is None or _model_fp16 != fp16:
            _model = None
            torch.cuda.empty_cache()
            print(f"[LOAD] Loading CosyVoice3 from {MODEL_DIR} (fp16={fp16})...", flush=True)
            _model = AutoModel(model_dir=MODEL_DIR, fp16=fp16)
            _model_fp16 = fp16
            print("[LOAD] Model ready.", flush=True)
    return _model


def handler(job):
    try:
        input_data = job.get("input", {})
        text = input_data.get("text", "")
        if not text:
            return {"ok": False, "error": "Missing 'text' in input."}

        emotion = input_data.get("emotion") or "natural"
        if emotion not in EMOTION_REFS:
            return {"ok": False, "error": f"Unknown emotion '{emotion}'. Valid: {sorted(EMOTION_REFS)}."}
        ref_wav, ref_text = EMOTION_REFS[emotion]

        # An explicit custom reference still overrides the emotion pick —
        # emotion selection only matters when the caller didn't supply one.
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

        cosy = get_model(fp16=bool(input_data.get("fp16", False)))
        prompt = DEFAULT_INSTRUCT + ref_text

        t0 = time.perf_counter()
        # Pre-split into safe-sized pieces (see chunk_text's docstring/comment
        # above), then run each through the model. CosyVoice3 may still
        # further subdivide any given piece internally and yields one output
        # per its own sub-piece too — ALL of those must be collected and
        # stitched together, not just the first (that was the original bug:
        # a plain `break` after the first yield silently discarded the rest).
        speech_pieces = []
        for piece in chunk_text(text):
            for out in cosy.inference_zero_shot(piece, prompt, ref_wav, stream=False, speed=speed):
                speech_pieces.append(out["tts_speech"])
        if not speech_pieces:
            return {"ok": False, "error": "Model produced no output."}
        speech = speech_pieces[0] if len(speech_pieces) == 1 else torch.cat(speech_pieces, dim=1)
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        buf = io.BytesIO()
        torchaudio.save(buf, speech, cosy.sample_rate, format="wav")
        wav_bytes = buf.getvalue()

        # RunPod's serverless result payload has a hard ~20MB ceiling — a few
        # minutes of raw 24kHz WAV lands right at that edge (confirmed: a
        # 351s generation's job completed but its output never came back).
        # Opus at 32kbps is comfortably smaller (~10-20x) for the same speech
        # at any realistic length, so transcode via the ffmpeg already baked
        # into this image rather than ship the raw WAV.
        audio_bytes, audio_format = wav_bytes, "wav"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "pipe:1"],
                input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            )
            if proc.returncode == 0 and proc.stdout:
                audio_bytes, audio_format = proc.stdout, "ogg"
            else:
                print("[WARN] ffmpeg opus encode failed, falling back to wav:", proc.stderr.decode(errors="replace")[-500:], flush=True)
        except Exception:
            print("[WARN] ffmpeg opus encode raised, falling back to wav:\n" + traceback.format_exc(), flush=True)

        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "ok": True,
            "audio_base64": audio_b64,
            "sample_rate": cosy.sample_rate,
            "format": audio_format,
            "inference_ms": inference_ms,
        }
    except Exception:
        print(traceback.format_exc(), flush=True)
        return {"ok": False, "error": "Internal server error"}


runpod.serverless.start({"handler": handler})
