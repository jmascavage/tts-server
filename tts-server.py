"""
tts-server.py  (latency-optimized)

Changes vs. the original:
  * No TensorFlow import (slow to load, unused).
  * Device auto-select (cuda / mps / cpu) with automatic CPU fallback if the model fails on MPS.
  * Warm-up synthesis at startup, so the first real request isn't slow.
  * Audio returned from memory (no temp files, which the original leaked).
  * Optional pitch/slow/reverb effects applied HERE (fast on the Mac) instead of on the Pi.
    Needs:  pip install pedalboard
    If pedalboard isn't installed the server still works and the client falls back to sox locally.
  * Lock around synthesis (the TTS model is not thread-safe) and a /health endpoint.
"""
import io
import threading
import time

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file
from TTS.api import TTS

try:
    from pedalboard import Pedalboard, PitchShift, Reverb
    HAVE_PEDALBOARD = True
except ImportError:
    HAVE_PEDALBOARD = False

MODEL_NAME = "tts_models/en/vctk/vits"
DEFAULT_SPEAKER = "p234"
SLOWDOWN = 0.95          # matches sox 'speed 0.95' in the original client chain
REVERB_TAIL_SECONDS = 0.5

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Preferred device: {device}   pedalboard effects: {HAVE_PEDALBOARD}")

tts = TTS(model_name=MODEL_NAME, progress_bar=False)
tts_lock = threading.Lock()


def warmup():
    tts.tts(text="Warming up the void.", speaker=DEFAULT_SPEAKER)


try:
    tts.to(device)
    warmup()
except Exception as e:                      # some VITS ops can fail on MPS
    print(f"Device '{device}' failed ({e}); falling back to CPU")
    device = "cpu"
    tts.to("cpu")
    warmup()

try:
    SAMPLE_RATE = int(tts.synthesizer.output_sample_rate)
except Exception:
    SAMPLE_RATE = 22050
print(f"Running on {device}, sample rate {SAMPLE_RATE}")


def apply_effects(wav, rate, pitch_semitones):
    """pitch down -> slow down 5% (resample, as sox 'speed' does) -> reverb with tail."""
    audio = np.asarray(wav, dtype=np.float32)
    audio = Pedalboard([PitchShift(semitones=pitch_semitones)])(audio, rate)

    n = int(len(audio) / SLOWDOWN)
    audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)

    audio = np.concatenate([audio, np.zeros(int(REVERB_TAIL_SECONDS * rate), dtype=np.float32)])
    audio = Pedalboard([Reverb(room_size=0.3, damping=0.5, wet_level=0.2, dry_level=0.8)])(audio, rate)
    return np.clip(audio, -1.0, 1.0)


app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify(status="ok", device=device, sample_rate=SAMPLE_RATE, effects=HAVE_PEDALBOARD)


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="no text"), 400

    speaker = data.get("speaker") or DEFAULT_SPEAKER
    want_effects = bool(data.get("effects", False))
    pitch = float(data.get("pitch_shift", -3.0))

    t0 = time.time()
    with tts_lock:
        wav = tts.tts(text=text, speaker=speaker)
    t_synth = time.time() - t0

    wav = np.asarray(wav, dtype=np.float32)
    applied = False
    if want_effects and HAVE_PEDALBOARD:
        try:
            wav = apply_effects(wav, SAMPLE_RATE, pitch)
            applied = True
        except Exception as e:
            print(f"Effects failed, sending raw audio: {e}")

    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    print(f"'{text[:50]}' synth={t_synth:.2f}s total={time.time() - t0:.2f}s effects={applied}")

    resp = send_file(buf, mimetype="audio/wav", as_attachment=False)
    resp.headers["X-Effects-Applied"] = "1" if applied else "0"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, threaded=True)
