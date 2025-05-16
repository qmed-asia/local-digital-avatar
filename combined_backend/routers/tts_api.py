
from fastapi import APIRouter, HTTPException, Request 
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Dict 
import os
import io
import sys
import wave
from uuid import uuid4

from pydantic import BaseModel
from piper.voice import PiperVoice
import combined_backend.config as config
from starlette.background import BackgroundTasks 


# Define the request body model
class ISynthesize(BaseModel):
    text: str
    length_scale: Optional[float] = config.PIPER_TTS_DEFAULT_LENGTH_SCALE 
    speaker: Optional[str] = config.PIPER_TTS_DEFAULT_SPEAKER 
    keep_file: Optional[bool] = False 


# Define the router for TTS endpoints
router = APIRouter(
    prefix="/tts",
    tags=["Text-to-Speech"]
    )



# Helper function to remove temporary files
async def remove_file(file_path: str):
    """Helper background task to remove a file after it's sent."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
             print(f"ERROR: Failed to remove file {file_path}: {e}", flush=True)


# Define the synthesis endpoint
@router.post("/speech")
async def synthesize_endpoint(data: ISynthesize, request: Request, bg_task: BackgroundTasks):
    """
    Endpoint to synthesize speech using the initialized Piper TTS pipeline.
    """
    # Access the loaded Piper voices from request.app.state
    loaded_voices: Dict[str, PiperVoice] = getattr(request.app.state, 'piper_tts_voices', {})

    # Check if the requested speaker is loaded
    if data.speaker not in loaded_voices:
        raise HTTPException(status_code=404, detail=f"Speaker '{data.speaker}' not found. Available speakers: {list(loaded_voices.keys())}")

    # Get the requested voice instance
    voice: PiperVoice = loaded_voices[data.speaker]

    try:
        wav_io = io.BytesIO()
        # Synthesize speech directly to the BytesIO object
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize(data.text, wav_file, length_scale=data.length_scale)

        print(f"DEBUG: Bytes written to wav_io: {wav_io.tell()}", flush=True)

        wav_io.seek(0) # Rewind the BytesIO object to the beginning

        if data.keep_file:
            save_directory = config.PIPER_TTS_DATA_DIR_FULL # Or config.TEMP_AUDIO_DIR if preferred for temp saves
            os.makedirs(save_directory, exist_ok=True) # Ensure directory exists

            filename = f"{uuid4()}.wav"
            output_file_path = os.path.join(save_directory, filename)

            with open(output_file_path, "wb") as f:
                f.write(wav_io.read())

            # Get duration for the JSON response
            try:
                # Need to reopen the saved file to get duration without affecting the stream
                with wave.open(output_file_path, "rb") as saved_wav_file:
                    frames = saved_wav_file.getnframes()
                    rate = saved_wav_file.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0
            except Exception as duration_error:
                 print(f"WARNING: Could not determine duration for saved file {output_file_path}: {duration_error}", flush=True)
                 duration = 0 # Default to 0 if duration cannot be read


            print(f"DEBUG: Saved synthesized audio to {output_file_path}", flush=True)
            return JSONResponse(content={"filename": filename, "duration": duration})

        else:
            def audio_file_bytes_generator():

                chunk_size = 4096 # Define a chunk size for streaming
                while True:
                    chunk = wav_io.read(chunk_size)
                    if not chunk:
                        break # End of BytesIO stream
                    yield chunk
            print("DEBUG: Returning StreamingResponse with audio generator.", flush=True)

            # If keep_file is False, stream the audio directly
            return StreamingResponse(audio_file_bytes_generator(), media_type="audio/wav")

    except Exception as e:
        # Log the error for debugging
        print(f"ERROR: Error during Piper TTS synthesis: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Failed to synthesize speech. Error: {e}")


