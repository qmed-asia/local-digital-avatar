# combined_backend/routers/avatar_api.py

import os
import io
import sys
import time
import asyncio
import wave
import tempfile
import shutil
import traceback
import base64

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict

# Import necessary classes/types from your services
from ..tts_service.tts_utils import initialize_piper_tts
from ..lipsync_service.overwrite.ov_wav2lip import OVWav2Lip

# Import config
import combined_backend.config as config

# --- Helper function for background file removal ---
async def remove_file(file_path: str):
    """
    Background task to safely remove a file after a delay.
    """
    await asyncio.sleep(config.BACKGROUND_CLEANUP_DELAY)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"DEBUG: Successfully removed background file: {file_path}", flush=True)
        except Exception as e:
             print(f"ERROR: Failed background file removal {file_path}: {e}", flush=True)


router = APIRouter(
    prefix="/avatar",
    tags=["Avatar"]
)

# --- Define Request Body Model ---

class TextToVideoRequest(BaseModel):
    """Request body for the text-to-video endpoint."""
    text: str
    tts_speaker: Optional[str] = config.PIPER_TTS_DEFAULT_SPEAKER
    tts_length_scale: Optional[float] = config.PIPER_TTS_DEFAULT_LENGTH_SCALE
    lipsync_reversed: Optional[bool] = False
    lipsync_starting_frame: Optional[int] = 0
    lipsync_enhance: Optional[bool] = True

# --- Helper to read file content asynchronously (in a thread) ---
async def read_file_content(file_path: str) -> bytes:
    """Reads the full content of a file in a thread pool."""
    # Use asyncio.to_thread for blocking file reading
    return await asyncio.to_thread(_read_file_sync, file_path)

# Synchronous helper for actual file reading
def _read_file_sync(file_path: str) -> bytes:
    """Synchronously reads the full content of a file."""
    with open(file_path, "rb") as f:
        return f.read()


# --- New Text-to-Video Endpoint ---

@router.post("/text-to-video")
async def text_to_video_endpoint_base64(data: TextToVideoRequest, request: Request, bg_task: BackgroundTasks):
    """
    Takes text, synthesizes audio using TTS, generates a talking avatar video,
    and returns the video file as Base64 encoded string in a JSON response.
    """
    start_time = time.time()
    print("DEBUG: Received /avatar/v1/text-to-video (Base64) request.", flush=True)
    print(f"DEBUG: Input text: {data.text[:100]}...", flush=True)
    print(f"DEBUG: TTS speaker: {data.tts_speaker}, length_scale: {data.tts_length_scale}", flush=True)
    print(f"DEBUG: Lipsync reversed: {data.lipsync_reversed}, starting_frame: {data.lipsync_starting_frame}, enhance: {data.lipsync_enhance}", flush=True)

    # --- 1. Access Initialized Services from app.state ---
    loaded_tts_voices: Dict[str, initialize_piper_tts] = getattr(request.app.state, 'piper_tts_voices', None)
    lipsync_instance: OVWav2Lip = getattr(request.app.state, 'wav2lip_pipeline', None)

    if loaded_tts_voices is None:
        print("ERROR: Piper TTS voices not loaded in app.state.", flush=True)
        raise HTTPException(status_code=503, detail="TTS service not fully initialized.")
    if lipsync_instance is None:
        print("ERROR: Lipsync pipeline not loaded in app.state.", flush=True)
        raise HTTPException(status_code=503, detail="Lipsync service not fully initialized.")

    # --- Validate TTS Speaker ---
    if data.tts_speaker not in loaded_tts_voices:
         available_speakers = list(loaded_tts_voices.keys())
         print(f"ERROR: Requested TTS speaker '{data.tts_speaker}' not loaded.", flush=True)
         raise HTTPException(status_code=404, detail=f"Speaker '{data.tts_speaker}' not found. Available speakers: {available_speakers}")

    tts_voice: initialize_piper_tts = loaded_tts_voices[data.tts_speaker]


    # --- 2. Synthesize Audio using TTS ---
    tts_start_time = time.time()
    audio_input_path = None
    temp_output_dir = config.TEMP_AUDIO_DIR
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        print("DEBUG: Starting TTS synthesis to temporary BytesIO.", flush=True)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
             tts_voice.synthesize(data.text, wav_file, length_scale=data.tts_length_scale)

        print(f"DEBUG: TTS synthesis to BytesIO complete. Bytes written: {wav_io.tell()}", flush=True)
        wav_io.seek(0)

        with tempfile.NamedTemporaryFile(prefix="tts_audio_", suffix=".wav", dir=temp_output_dir, delete=False) as temp_audio_file:
             audio_input_path = temp_audio_file.name
             temp_audio_file.write(wav_io.read())

        print(f"DEBUG: Audio saved to temporary file: {audio_input_path}", flush=True)
        wav_io.close()

        tts_end_time = time.time()
        print(f"INFO: TTS synthesis and temp file save took {tts_end_time - tts_start_time:.2f} seconds.", flush=True)


    except Exception as e:
        # ... (TTS error handling remains the same) ...
        print(f"ERROR: TTS synthesis or saving to temporary file failed: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        if audio_input_path and os.path.exists(audio_input_path):
             os.remove(audio_input_path)
        raise HTTPException(status_code=500, detail=f"Audio synthesis failed: {e}")


    # --- 3. Perform Lipsync Inference ---
    lipsync_start_time = time.time()
    final_video_path = None
    try:
        os.makedirs(config.LIPSYNC_RESULTS_DIR_FULL, exist_ok=True)
        video_output_filename = f"lipsync_result_{os.getpid()}_{os.urandom(8).hex()}.mp4"


        print(f"DEBUG: Starting Lipsync inference with audio file {audio_input_path}.", flush=True)

        result_file_id = await asyncio.to_thread(
            lipsync_instance.inference,
            audio_input_path,
            reversed=data.lipsync_reversed,
            starting_frame=data.lipsync_starting_frame,
            enhance=data.lipsync_enhance,
            output_dir=config.LIPSYNC_RESULTS_DIR_FULL
        )

        lipsync_end_time = time.time()
        print(f"INFO: Lipsync inference took {lipsync_end_time - lipsync_start_time:.2f} seconds.", flush=True)


        print(f"DEBUG: Lipsync inference completed. Returned ID: {result_file_id}", flush=True)

        final_video_path = os.path.join(config.LIPSYNC_RESULTS_DIR_FULL, f"{result_file_id}.mp4")

        if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
             print(f"ERROR: Lipsync inference completed but expected output file not found or empty: {final_video_path}", flush=True)
             if audio_input_path and os.path.exists(audio_input_path):
                 os.remove(audio_input_path)
             raise RuntimeError("Lipsync inference failed to produce output file.")


        print(f"DEBUG: Lipsync output file found: {final_video_path}. Reading for Base64 encoding.", flush=True)

        # --- 4. Read Video File Content and Encode as Base64 ---
        video_content_bytes = await read_file_content(final_video_path)
        print(f"DEBUG: Read {len(video_content_bytes)} bytes from video file.", flush=True)

        video_base64 = base64.b64encode(video_content_bytes).decode('ascii')
        print(f"DEBUG: Video content encoded to Base64 (length: {len(video_base64)}).", flush=True)


    except Exception as e:
        print(f"ERROR: Lipsync inference or Base64 encoding failed: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        if audio_input_path and os.path.exists(audio_input_path):
             os.remove(audio_input_path)

        if final_video_path and os.path.exists(final_video_path):
             try:
                 os.remove(final_video_path)
                 print(f"DEBUG: Cleaned up video output file after error: {final_video_path}", flush=True)
             except Exception as cleanup_e:
                  print(f"WARNING: Failed to clean up video file {final_video_path}: {cleanup_e}", flush=True)

        raise HTTPException(status_code=500, detail=f"Video generation or processing failed: {e}")

    finally:
        if audio_input_path and os.path.exists(audio_input_path):
             try:
                 os.remove(audio_input_path)
                 print(f"DEBUG: Cleaned up temporary audio file: {audio_input_path}", flush=True)
             except Exception as cleanup_e:
                  print(f"WARNING: Failed to clean up temporary audio file {audio_input_path}: {cleanup_e}", flush=True)


    # --- 5. Return JSON Response with Base64 Encoded Video ---
    if final_video_path and os.path.exists(final_video_path):
         bg_task.add_task(remove_file, final_video_path)
         print(f"DEBUG: Background cleanup task added for video file: {final_video_path}", flush=True)
    else:
         print("WARNING: No final video path available to add background cleanup task.", flush=True)


    print("DEBUG: Returning JSONResponse with Base64 encoded video.", flush=True)
    end_time = time.time()
    print(f"INFO: Text-to-Video complete in {end_time - start_time:.2f} seconds (Total).", flush=True)
    return JSONResponse(content={"video_base64": video_base64, "message": "Video generated successfully"})

