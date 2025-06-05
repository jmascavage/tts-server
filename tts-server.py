from flask import Flask, request, send_file
from TTS.api import TTS
import soundfile as sf
import tempfile
import os
import torch

if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("MPS backend is not available.")

import tensorflow as tf
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)    

mps_device = torch.device("mps")
'''
# Get device
device = "cuda" if torch.cuda.is_available() else 'mps' # "mps"
# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
'''

app = Flask(__name__)
#tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True)  # Use gpu=False if CPU only
tts = TTS(model_name="tts_models/en/vctk/vits").to(mps_device)  # on Macs use MPS since GPU=nvidia=not-on=-ac


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")
    speaker_id = data.get("speaker_id", 10)

    # Generate audio
#    wav = tts.tts(text=text, speaker_idx=speaker_id)
    wav = tts.tts(text=text, speaker='p234')
    tmp_path = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_path, wav, samplerate=22050)

    return send_file(tmp_path, mimetype="audio/wav", as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)