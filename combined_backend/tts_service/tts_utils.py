
import os
from urllib.request import urlretrieve
import wave
import io
from typing import Dict

from piper.voice import PiperVoice

import combined_backend.config as config

import logging
logger = logging.getLogger("uvicorn.error") # Use the same logger name as app.py


def initialize_piper_tts(app_state: object):
    """
    Initializes the Piper TTS service by downloading models if needed
    and loading them into the application state.
    """
    print("DEBUG: Initializing Piper TTS service (from tts_utils)...", flush=True)
    # Initialize an empty dictionary to store loaded Piper voices
    app_state.piper_tts_voices = {}

    # Ensure the Piper TTS models directory exists
    if not os.path.exists(config.PIPER_TTS_MODELS_DIR_FULL):
        os.makedirs(config.PIPER_TTS_MODELS_DIR_FULL, exist_ok=True)
        print(f"DEBUG: Created Piper TTS models directory: {config.PIPER_TTS_MODELS_DIR_FULL}", flush=True)


    # Loop through the defined speakers and download/load models
    for speaker_name, model_path_relative in config.PIPER_TTS_SPEAKER_MODEL_PATHS.items():
        model_name = model_path_relative.split("/")[-1]

        model_file_path = os.path.join(config.PIPER_TTS_MODELS_DIR_FULL, f"{model_name}.onnx")
        model_config_file_path = os.path.join(config.PIPER_TTS_MODELS_DIR_FULL, f"{model_name}.onnx.json")

        # Download the model file if it doesn't exist
        if not os.path.exists(model_file_path):
            model_download_url = f"{config.PIPER_TTS_DOWNLOAD_URL}/{model_path_relative}.onnx"
            print(f"DEBUG: Downloading Piper TTS model for '{speaker_name}': {model_download_url} to {model_file_path}", flush=True)
            urlretrieve(model_download_url, model_file_path) # nosec --http file
            print(f"DEBUG: Downloaded {model_file_path}", flush=True)

        # Download the model config file if it doesn't exist
        if not os.path.exists(model_config_file_path):
            model_config_url = f"{config.PIPER_TTS_DOWNLOAD_URL}/{model_path_relative}.onnx.json"
            print(f"DEBUG: Downloading Piper TTS config for '{speaker_name}': {model_config_url} to {model_config_file_path}", flush=True)
            urlretrieve(model_config_url, model_config_file_path) # nosec --http file
            print(f"DEBUG: Downloaded {model_config_file_path}", flush=True)

        # Load the Piper voice and store it in app.state
        print(f"DEBUG: Loading Piper TTS voice for '{speaker_name}' from {model_file_path}", flush=True)
        # PiperVoice.load_from_dir requires the directory, not the full .onnx path
        voice_directory = os.path.dirname(model_file_path)
        # PiperVoice.load_from_dir expects the *base* name without .onnx or .onnx.json
        voice_base_name = os.path.splitext(os.path.basename(model_file_path))[0]

        # Pass the directory containing the .onnx and .onnx.json, and the base name
        app_state.piper_tts_voices[speaker_name] = PiperVoice.load(model_file_path)
        print(f"DEBUG: Loaded Piper TTS voice for '{speaker_name}'.", flush=True)

    # Check if any voices were loaded
    if not app_state.piper_tts_voices:
         raise RuntimeError("No Piper TTS voices were loaded.")

    print("DEBUG: Piper TTS service initialization complete (from tts_utils).", flush=True)
