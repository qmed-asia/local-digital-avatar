# combined_backend/routers/avatar_streaming_api.py

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
import json

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse 

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
            print(f"DEBUG: [SSE Cleanup] Successfully removed background file: {file_path}", flush=True)
        except Exception as e:
             print(f"ERROR: [SSE Cleanup] Failed background file removal {file_path}: {e}", flush=True)


router = APIRouter(
    prefix="/avatar-sse",
    tags=["Avatar (SSE)"],
)

# --- Define Request Body Model ---
class TextToVideoRequest(BaseModel):
    """Request body for the text-to-video SSE endpoint."""
    text: str
    tts_speaker: Optional[str] = config.PIPER_TTS_DEFAULT_SPEAKER
    tts_length_scale: Optional[float] = config.PIPER_TTS_DEFAULT_LENGTH_SCALE
    lipsync_reversed: Optional[bool] = False
    lipsync_starting_frame: Optional[int] = 0
    lipsync_enhance: Optional[bool] = True


# --- Main Generator Function for SSE Stream ---
async def generate_avatar_sse_stream(
    data: TextToVideoRequest,
    app_state: object,
    bg_task: BackgroundTasks
):
    start_time = time.time()
    # print("DEBUG: [SSE] Starting generate_avatar_sse_stream generator.", flush=True)

    # --- Initial Messages ---
    # Yield initial status messages formatted as SSE data
    # yield f"data: {json.dumps({'status': 'initialized', 'message': 'Starting video generation process'})}\n\n"
    # yield f"data: {json.dumps({'status': 'processing', 'message': 'Accessing services...'})}\n\n"

    # --- 1. Access Initialized Services from app.state ---
    loaded_tts_voices: Dict[str, initialize_piper_tts] = getattr(app_state, 'piper_tts_voices', None)
    lipsync_instance: OVWav2Lip = getattr(app_state, 'wav2lip_pipeline', None)

    if loaded_tts_voices is None or lipsync_instance is None:
         error_msg = "Required services not fully initialized. Please check server logs."
         print(f"ERROR: [SSE] {error_msg}", flush=True)
         yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"
         return

    # --- Validate TTS Speaker ---
    if data.tts_speaker not in loaded_tts_voices:
         available_speakers = list(loaded_tts_voices.keys())
         error_msg = f"TTS speaker '{data.tts_speaker}' not found. Available speakers: {available_speakers}"
         print(f"ERROR: [SSE] {error_msg}", flush=True)
         yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"
         return

    tts_voice: initialize_piper_tts = loaded_tts_voices[data.tts_speaker]

    # --- 2. Synthesize Audio using TTS ---
    tts_start_time = time.time()
    audio_input_path = None
    temp_output_dir = config.TEMP_AUDIO_DIR
    os.makedirs(temp_output_dir, exist_ok=True)

    yield f"data: {json.dumps({'status': 'processing', 'message': 'Synthesizing audio...'})}\n\n"

    try:
        print("DEBUG: [SSE] Starting TTS synthesis to temporary BytesIO.", flush=True)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
             tts_voice.synthesize(data.text, wav_file, length_scale=data.tts_length_scale)

        print(f"DEBUG: [SSE] TTS synthesis to BytesIO complete. Bytes written: {wav_io.tell()}", flush=True)
        wav_io.seek(0)

        with tempfile.NamedTemporaryFile(prefix="tts_audio_", suffix=".wav", dir=temp_output_dir, delete=False) as temp_audio_file:
             audio_input_path = temp_audio_file.name
             temp_audio_file.write(wav_io.read())

        print(f"DEBUG: [SSE] Audio saved to temporary file: {audio_input_path}", flush=True)
        wav_io.close()

        tts_end_time = time.time()
        tts_duration = tts_end_time - tts_start_time
        print(f"INFO: [SSE] TTS synthesis and temp file save took {tts_duration:.2f} seconds.", flush=True)
        yield f"data: {json.dumps({'status': 'processing', 'message': f'Audio synthesis complete in {tts_duration:.2f}s'})}\n\n"

    except Exception as e:
        error_msg = f"Audio synthesis failed: {e}"
        print(f"ERROR: [SSE] {error_msg}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        # Clean up temporary audio file if path was assigned
        if audio_input_path and os.path.exists(audio_input_path):
             os.remove(audio_input_path)
        yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"
        return

    # --- 3. Perform Lipsync Inference ---
    lipsync_start_time = time.time()
    final_video_path = None

    yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting lipsync inference...'})}\n\n"

    try:
        os.makedirs(config.LIPSYNC_RESULTS_DIR_FULL, exist_ok=True)
        video_output_filename = f"lipsync_result_{os.getpid()}_{os.urandom(8).hex()}.mp4"

        final_video_path = os.path.join(config.LIPSYNC_RESULTS_DIR_FULL, video_output_filename)

        print(f"DEBUG: [SSE] Starting Lipsync inference with audio file {audio_input_path}.", flush=True)

        result_file_id = await asyncio.to_thread(
            lipsync_instance.inference,
            audio_input_path,
            reversed=data.lipsync_reversed,
            starting_frame=data.lipsync_starting_frame,
            enhance=data.lipsync_enhance,
            output_dir=config.LIPSYNC_RESULTS_DIR_FULL
        )

        lipsync_end_time = time.time()
        lipsync_duration = lipsync_end_time - lipsync_start_time
        print(f"INFO: [SSE] Lipsync inference took {lipsync_duration:.2f} seconds.", flush=True)
        yield f"data: {json.dumps({'status': 'processing', 'message': f'Lipsync inference complete in {lipsync_duration:.2f}s'})}\n\n"

        print(f"DEBUG: [SSE] Lipsync inference completed. Returned ID: {result_file_id}", flush=True)

        # Verify the file was created by the inference method using the returned ID
        actual_generated_path = os.path.join(config.LIPSYNC_RESULTS_DIR_FULL, f"{result_file_id}.mp4")
        if final_video_path != actual_generated_path:
            print(f"WARNING: [SSE] Constructed final_video_path {final_video_path} does not match actual generated path {actual_generated_path}. Using actual.", flush=True)
            final_video_path = actual_generated_path

        if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
             error_msg = f"Lipsync inference failed to produce output file or produced an empty file: {final_video_path}"
             print(f"ERROR: [SSE] {error_msg}", flush=True)
             yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"
             # Clean up temporary audio file and potential partial video on this specific failure path
             if audio_input_path and os.path.exists(audio_input_path): os.remove(audio_input_path)
             if final_video_path and os.path.exists(final_video_path): os.remove(final_video_path)
             return

        print(f"DEBUG: [SSE] Lipsync output file found: {final_video_path}. Starting to stream Base64 chunks.", flush=True)
        yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting to stream video chunks...'})}\n\n"


        # --- 4. Read Video File Content and Stream as Base64 Chunks (like video_file_generator_base64) ---
        chunk_size = 819200 # You can adjust this size as needed
        chunk_count = 0 

        try:
            with open(final_video_path, "rb") as f:
                 print(f"DEBUG: [SSE Streaming] Starting to read and encode chunks...", flush=True)
                 while True:
                      chunk = f.read(chunk_size)
                      if not chunk:
                           print(f"DEBUG: [SSE Streaming] Finished reading file.", flush=True)
                           break

                      chunk_count += 1
                      # Encode the binary chunk to Base64 text and decode to string
                      base64_chunk = base64.b64encode(chunk).decode('ascii')

                      # Yield a message containing the Base64 chunk
                      yield f"data: {json.dumps({'status': 'streaming', 'chunk_number': chunk_count, 'video_chunk_base64': base64_chunk})}\n\n"

            # --- Add background task to remove the file AFTER the generator has finished reading ---
            bg_task.add_task(remove_file, final_video_path)
            print(f"DEBUG: [SSE Cleanup] Background cleanup task added for video file: {final_video_path}", flush=True)


        except Exception as stream_e:
            error_msg = f"Error during video streaming: {stream_e}"
            print(f"ERROR: [SSE Streaming] {error_msg}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"

    except Exception as e:
        error_msg = f"Lipsync inference failed: {e}"
        print(f"ERROR: [SSE] {error_msg}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"


    finally:
        # --- Clean up temporary files ---
        # Clean up the temporary audio file
        if audio_input_path and os.path.exists(audio_input_path):
             try:
                 os.remove(audio_input_path)
                 # print(f"DEBUG: [SSE Cleanup] Cleaned up temporary audio file: {audio_input_path}", flush=True)
             except Exception as cleanup_e:
                  print(f"WARNING: [SSE Cleanup] Failed to clean up temporary audio file {audio_input_path}: {cleanup_e}", flush=True)

        if final_video_path and os.path.exists(final_video_path):
             pass


    # --- Final Success Message ---
    if 'error_msg' not in locals() and final_video_path and os.path.exists(final_video_path):
        end_time = time.time()
        total_duration = end_time - start_time
        # print(f"INFO: [SSE] Generate Avatar SSE complete in {total_duration:.2f} seconds (Total).", flush=True)
        yield f"data: {json.dumps({'status': 'completed', 'message': f'Video generation complete in {total_duration:.2f}s'})}\n\n"
        # print("DEBUG: [SSE] Generator finished yielding successfully.", flush=True)
    elif 'error_msg' not in locals():
         print("WARNING: [SSE] Generator finished without explicit error but no completion message sent.", flush=True)

    # print("DEBUG: [SSE] Generator function exiting.", flush=True)


# --- Endpoint Definition ---
@router.post("/text-to-video")
async def text_to_video_sse_endpoint(
    data: TextToVideoRequest,
    request: Request,
    bg_task: BackgroundTasks
):
    """
    Takes text, synthesizes audio using TTS, generates a talking avatar video,
    and streams progress and Base64 encoded chunks as Server-Sent Events (SSE).
    """
    print("DEBUG: [SSE Endpoint] Request received, returning StreamingResponse.", flush=True)
    # Return a StreamingResponse using the generator
    return StreamingResponse(
        generate_avatar_sse_stream(data, request.app.state, bg_task),
        media_type="text/event-stream" 
    )
