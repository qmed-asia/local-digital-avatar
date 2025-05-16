
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import shutil
import tempfile
import time
import sys
import os
import io
import asyncio

import combined_backend.config as config

from ..lipsync_service.overwrite.ov_wav2lip import OVWav2Lip

# --- Helper function for background file removal ---
async def remove_file(file_name):
    """
    Background task to safely remove a file.
    """
    # Add a small delay to ensure the file is no longer being read/streamed
    await asyncio.sleep(15)
    if os.path.exists(file_name):
        try:
            os.remove(file_name)
            # print(f"DEBUG: Successfully removed file: {file_name}", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to remove file {file_name}: {e}", flush=True)


router = APIRouter(
    prefix="/lipsync",
    tags=["Lipsync"],
)

# Define request models if needed (e.g., for /inference_from_filename)
class InferenceFromFilenameRequest(BaseModel):
    audio_filename: str
    reversed: bool = False
    starting_frame: int = 0
    enhance: bool = True


@router.post("/inference")
async def inference_upload(
    request: Request,
    bg_task: BackgroundTasks,
    starting_frame: int = 0, 
    reversed: bool = False, 
    enhance: bool = True, 
    file: UploadFile = File(...)
):
    """
    Performs lipsync inference on an uploaded audio file.
    """
    # print(f"DEBUG: Received /lipsync/v1/inference request for file: {file.filename}", flush=True)

    # Access the initialized pipeline from app state
    lipsync_pipeline: OVWav2Lip = request.app.state.wav2lip_pipeline

    if lipsync_pipeline is None:
         print("ERROR: Lipsync pipeline not initialized.", flush=True)
         raise HTTPException(status_code=503, detail="Lipsync service not initialized")

    if not file.filename.lower().endswith(".wav"): # Use lower() for case-insensitivity
        print(f"ERROR: Invalid file type uploaded: {file.filename}", flush=True)
        raise HTTPException(status_code=400, detail="Only .wav files are allowed")

    # Ensure the results directory exists
    os.makedirs(config.LIPSYNC_RESULTS_DIR_FULL, exist_ok=True)

    temp_audio_filename = f"temp_audio_{os.getpid()}_{os.urandom(8).hex()}.wav"
    temp_audio_path = os.path.join(config.DATA_DIRECTORY, temp_audio_filename)

    print(f"DEBUG: Saving uploaded audio to temporary path: {temp_audio_path}", flush=True)
    try:
        # Using standard open/write pattern compatible with background task cleanup
        with open(temp_audio_path, "wb") as f:
             shutil.copyfileobj(file.file, f)
        print("DEBUG: Audio file saved temporarily.", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to save temporary audio file: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to process audio upload: {e}")


    # Call lipsync_pipeline.inference
    print(f"DEBUG: Calling lipsync_pipeline.inference with starting_frame={starting_frame}, reversed={reversed}, enhance={enhance}", flush=True)
    try:
        # The inference method should return the generated result file ID (without extension)
        # or the full path to the result video file. Let's assume it returns the ID.
        # Original code seemed to return just the ID.
        result_file_id = lipsync_pipeline.inference(
             temp_audio_path,
             reversed=reversed,
             starting_frame=starting_frame,
             enhance=enhance, 
             output_dir=config.LIPSYNC_RESULTS_DIR_FULL
        )
        print(f"DEBUG: Inference complete. Result file ID: {result_file_id}", flush=True)

        # Construct the full path to the result video file
        result_video_path = os.path.join(config.LIPSYNC_RESULTS_DIR_FULL, f"{result_file_id}.mp4")
        print(f"DEBUG: Expected result video path: {result_video_path}", flush=True)

        # Check if the result video file was actually created by the inference method
        if not os.path.exists(result_video_path):
            print(f"ERROR: Inference completed but result file not found at {result_video_path}", flush=True)
            raise HTTPException(status_code=500, detail="Inference failed to produce output file.")

        # Return the video file as a FileResponse
        # Add background tasks to clean up the temporary audio file and the generated video file
        bg_task.add_task(remove_file, temp_audio_path) 
        bg_task.add_task(remove_file, result_video_path) 

        print(f"DEBUG: Returning result video {result_video_path} and adding cleanup tasks.", flush=True)
        return FileResponse(result_video_path, media_type="video/mp4", background=bg_task)

    except Exception as e:
        print(f"ERROR: An error occurred during inference: {e}", flush=True)
        # Attempt to clean up temp audio file if it exists
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"DEBUG: Cleaned up temp audio file after error: {temp_audio_path}", flush=True)
            except: pass
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@router.get("/test")
async def test_inference(request: Request, enhance: bool = True): # Renamed to test_inference for clarity
    """
    Performs a test lipsync inference using a default audio file.
    """
    # print(f"DEBUG: Received /lipsync/v1/test request with enhance={enhance}", flush=True)

    lipsync_pipeline: OVWav2Lip = request.app.state.wav2lip_pipeline

    if lipsync_pipeline is None:
         print("ERROR: Lipsync pipeline not initialized for /test.", flush=True)
         raise HTTPException(status_code=503, detail="Lipsync service not initialized")

    # Use a hardcoded path to a test audio file within the project structure or data directory
    test_audio_path = os.path.join(config.PROJECT_ROOT, "assets", "audio.wav") # Make sure assets/audio.wav exists

    if not os.path.exists(test_audio_path):
         print(f"ERROR: Test audio file not found at {test_audio_path}.", flush=True)
         raise HTTPException(status_code=404, detail=f"Test audio file not found at {test_audio_path}")


    print(f"DEBUG: Calling lipsync_pipeline.inference for test with audio {test_audio_path}", flush=True)
    start_time = time.time()
    try:
        # Use default parameters for reversed and starting_frame for the test
        result_file_id = lipsync_pipeline.inference(test_audio_path, reversed=False, starting_frame=0, enhance=enhance, output_dir=config.LIPSYNC_RESULTS_DIR_FULL)
        end_time = time.time()
        print(f"DEBUG: Test inference took {end_time - start_time} seconds. Result ID: {result_file_id}", flush=True)

        result_video_url = request.url_for("get_video_result", file_id=result_file_id)

        return JSONResponse(content={"message": "Test inference completed", "result_url": str(result_video_url)}, status_code=200)

    except Exception as e:
        print(f"ERROR: An error occurred during test inference: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Test inference failed: {e}")


@router.post("/inference_from_filename")
async def inference_from_filename(request: Request, req: InferenceFromFilenameRequest):
    """
    Performs lipsync inference using a pre-existing audio file specified by filename.
    Expects JSON body: {"audio_filename": "your_file.wav", "reversed": false, "starting_frame": 0, "enhance": false}
    """
    # print(f"DEBUG: Received /lipsync/v1/inference_from_filename request for file: {req.audio_filename}", flush=True)

    lipsync_pipeline: OVWav2Lip = request.app.state.wav2lip_pipeline

    if lipsync_pipeline is None:
         print("ERROR: Lipsync pipeline not initialized for /inference_from_filename.", flush=True)
         raise HTTPException(status_code=503, detail="Lipsync service not initialized")

    if not req.audio_filename.lower().endswith(".wav"): # Use lower() for case-insensitivity
        print(f"ERROR: Invalid filename extension provided: {req.audio_filename}", flush=True)
        raise HTTPException(status_code=400, detail="Only .wav filenames are allowed")

    # Construct the full absolute path to the audio file using config.DATA_DIRECTORY
    # Assuming the file is located directly within the DATA_DIRECTORY
    full_audio_path = os.path.join(config.DATA_DIRECTORY, req.audio_filename)

    print(f"DEBUG: Checking for audio file at: {full_audio_path}", flush=True)
    if not os.path.exists(full_audio_path):
        print(f"ERROR: Audio file not found at {full_audio_path}.", flush=True)
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.audio_filename}")


    print(f"DEBUG: Calling lipsync_pipeline.inference with file path: {full_audio_path}", flush=True)
    start_time = time.time()
    try:
        # Call lipsync_pipeline.inference with the full file path and parameters from req
        result_file_id = lipsync_pipeline.inference(
            full_audio_path,
            reversed=req.reversed,
            starting_frame=req.starting_frame,
            enhance=req.enhance,
            output_dir=config.LIPSYNC_RESULTS_DIR_FULL
        )
        end_time = time.time()
        print(f"DEBUG: Inference from filename took {end_time - start_time} seconds. Result ID: {result_file_id}", flush=True)

        # Construct the URL to access the video via the /video/{id} endpoint
        result_video_url = request.url_for("get_video_result", file_id=result_file_id)

        # Return JSON with the URL to the result video
        return JSONResponse(content={"message": "Inference completed", "result_url": str(result_video_url)}, status_code=200)

    except Exception as e:
        print(f"ERROR: An error occurred during inference from filename: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout) # Print traceback for errors
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@router.get("/video/{file_id}")
async def get_video_result(file_id: str, bg_task: BackgroundTasks):
    """
    Retrieves a previously generated lipsync video result by ID.
    Adds a background task to clean up the video file after streaming.
    """
    print(f"DEBUG: Received /lipsync/v1/video/{file_id} request.", flush=True)

    # Construct the full absolute path to the video file using file_id and config.LIPSYNC_RESULTS_DIR_FULL
    video_path = os.path.join(config.LIPSYNC_RESULTS_DIR_FULL, f"{file_id}.mp4")

    print(f"DEBUG: Looking for video file at: {video_path}", flush=True)
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}.", flush=True)
        raise HTTPException(status_code=404, detail="Video result not found")

    print(f"DEBUG: Found video file. Returning as StreamingResponse.", flush=True)
    response = FileResponse(video_path, media_type="video/mp4")

    # Add a background task to clean up the video file after it has been sent
    bg_task.add_task(remove_file, video_path) # Clean up video after sending

    print("DEBUG: Background task added for video file cleanup.", flush=True)
    return response # Return FileResponse