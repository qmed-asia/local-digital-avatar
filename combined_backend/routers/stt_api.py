
from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import os
import sys
import uuid
import shutil
import logging
from pydub import AudioSegment
import traceback

from combined_backend.stt_service.stt_utils import transcribe, translate

import combined_backend.config as config


logger = logging.getLogger("uvicorn.error")

# Define the router for STT endpoints
router = APIRouter(
    prefix="/stt",
    tags=["Speech-to-Text"]
    )


# Define the transcription endpoint
@router.post("/transcriptions")
async def stt_transcription_endpoint(request: Request, file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Endpoint to transcribe audio input using the initialized STT pipeline.
    """
    # Access the STT pipeline from request.app.state
    if not hasattr(request.app.state, 'asr_pipeline') or request.app.state.asr_pipeline is None:
        logger.error("STT pipeline is not initialized when transcription endpoint was called.")
        raise HTTPException(status_code=503, detail="STT service is not initialized yet.")

    # Use the temporary directory path from the config module
    temp_audio_dir_full = config.TEMP_AUDIO_DIR # Use path from config
    os.makedirs(temp_audio_dir_full, exist_ok=True)


    try:
        if file.filename:
            safe_filename = os.path.basename(file.filename)
            file_base_name = os.path.splitext(safe_filename)[0]
        else:
            file_base_name = str(uuid.uuid4())

        input_file_path_webm = os.path.join(temp_audio_dir_full, f"{file_base_name}.webm")
        input_file_path_wav = os.path.join(temp_audio_dir_full, f"{file_base_name}.wav")


        print(f"DEBUG: Saving uploaded file to {input_file_path_webm}", flush=True)
        with open(input_file_path_webm, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        print("DEBUG: Uploaded file saved.", flush=True)


        print(f"DEBUG: Converting webm to wav: {input_file_path_webm} -> {input_file_path_wav}", flush=True)
        try:
            audio = AudioSegment.from_file(input_file_path_webm, format="webm")
            audio.export(input_file_path_wav, format="wav")
            print("DEBUG: Audio conversion complete.", flush=True)
        except Exception as conversion_error:
             print(f"ERROR: Audio conversion failed: {conversion_error}", flush=True)
             if os.path.exists(input_file_path_webm): os.remove(input_file_path_webm)
             raise HTTPException(status_code=400, detail=f"Failed to convert audio file. Ensure it's a valid webm format. Error: {conversion_error}")


        if language is None:
            logger.warning("Language is not set for transcription. Defaulting to english.")
            language_to_use = "english"
        else:
            language_to_use = language

        print(f"DEBUG: Transcribing audio with language: {language_to_use} from {input_file_path_wav}", flush=True)
        # Call the transcribe function using the pipeline from request.app.state
        text = transcribe(
            pipeline=request.app.state.asr_pipeline, # Use request.app.state.asr_pipeline
            audio=input_file_path_wav,
            language=language_to_use
        )
        print("DEBUG: Transcription complete.", flush=True)


        print(f"DEBUG: Cleaning up temporary files: {input_file_path_webm}, {input_file_path_wav}", flush=True)
        if os.path.exists(input_file_path_webm):
            os.remove(input_file_path_webm)
        if os.path.exists(input_file_path_wav):
            os.remove(input_file_path_wav)
        print("DEBUG: Temporary files cleaned up.", flush=True)


        return {"text": text, 'status': True}

    except HTTPException as e:
        if os.path.exists(input_file_path_webm): os.remove(input_file_path_webm)
        if os.path.exists(input_file_path_wav) and input_file_path_wav != input_file_path_webm: os.remove(input_file_path_wav)
        raise e
    except Exception as error:
        logger.error(f"Error in STT transcriptions: {str(error)}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        if os.path.exists(input_file_path_webm): os.remove(input_file_path_webm)
        if os.path.exists(input_file_path_wav) and input_file_path_wav != input_file_path_webm: os.remove(input_file_path_wav)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe the voice input. Error: {error}"
        )


# Define the translation endpoint
@router.post("/translations")
async def stt_translation_endpoint(request: Request, file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Endpoint to translate audio input using the initialized STT pipeline.
    """
    # Access the STT pipeline from request.app.state
    if not hasattr(request.app.state, 'asr_pipeline') or request.app.state.asr_pipeline is None:
        logger.error("STT pipeline is not initialized when translation endpoint was called.")
        raise HTTPException(status_code=503, detail="STT service is not initialized yet.")

    # Use the temporary directory path from the config module
    temp_audio_dir_full = config.TEMP_AUDIO_DIR # Use path from config
    os.makedirs(temp_audio_dir_full, exist_ok=True)

    try:
        if file.filename:
            safe_filename = os.path.basename(file.filename)
            file_base_name = os.path.splitext(safe_filename)[0]
        else:
            file_base_name = str(uuid.uuid4())

        input_file_path_webm = os.path.join(temp_audio_dir_full, f"{file_base_name}.webm")
        input_file_path_wav = os.path.join(temp_audio_dir_full, f"{file_base_name}.wav")

        print(f"DEBUG: Saving uploaded file to {input_file_path_webm}", flush=True)
        with open(input_file_path_webm, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        print("DEBUG: Uploaded file saved.", flush=True)

        print(f"DEBUG: Converting webm to wav: {input_file_path_webm} -> {input_file_path_wav}", flush=True)
        try:
            audio = AudioSegment.from_file(input_file_path_webm, format="webm")
            audio.export(input_file_path_wav, format="wav")
            print("DEBUG: Audio conversion complete.", flush=True)
        except Exception as conversion_error:
             print(f"ERROR: Audio conversion failed: {conversion_error}", flush=True)
             if os.path.exists(input_file_path_webm): os.remove(input_file_path_webm)
             raise HTTPException(status_code=400, detail=f"Failed to convert audio file. Ensure it's a valid webm format. Error: {conversion_error}")


        if language is None:
            logger.warning("Source language is not set for translation. Defaulting to english.")
            source_language_to_use = "english"
        else:
             source_language_to_use = language


        print(f"DEBUG: Translating audio with source language: {source_language_to_use} from {input_file_path_wav}", flush=True)
        # Call the translate function using the global pipeline
        text = translate(
            pipeline=request.app.state.asr_pipeline, # Use request.app.state.asr_pipeline
            audio=input_file_path_wav,
            source_language=source_language_to_use
        )
        print("DEBUG: Translation complete.", flush=True)

        print(f"DEBUG: Cleaning up temporary files: {input_file_path_webm}, {input_file_path_wav}", flush=True)
        if os.path.exists(input_file_path_webm):
            os.remove(input_file_path_webm)
        if os.path.exists(input_file_path_wav):
            os.remove(input_file_path_wav)
        print("DEBUG: Temporary files cleaned up.", flush=True)

        return {"text": text, 'status': True}

    except HTTPException as e:
        if os.path.exists(input_file_path_webm): os.remove(input_file_path_webm)
        if os.path.exists(input_file_path_wav) and input_file_path_wav != input_file_path_webm: os.remove(input_file_path_wav)
        raise e
    except Exception as error:
        logger.error(f"Error in STT translations: {str(error)}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        if os.path.exists(input_file_path_webm): os.remove(input_file_path_webm)
        if os.path.exists(input_file_path_wav) and input_file_path_wav != input_file_path_webm: os.remove(input_file_path_wav)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to translate the voice input. Error: {error}"
        )
    
