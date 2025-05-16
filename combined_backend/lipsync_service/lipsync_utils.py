
import os
import sys
import time
import numpy as np
import cv2
import wave
import json
import hashlib
import traceback

import openvino as ov
import torch

from .overwrite.ov_wav2lip import OVWav2Lip, OVFaceAlignment, LandmarksType
from .realesrgan_overwrite.inference import initialize as initialize_enhancer

import combined_backend.config as config
from .hparams import hparams as hp

import logging
logger = logging.getLogger("uvicorn.error")

# --- Cache Helper Functions ---

def _calculate_file_hash(filepath: str, hash_algorithm='sha256'):
    """Calculates the hash of a file's content."""
    hasher = hashlib.new(hash_algorithm)
    try:
        with open(filepath, 'rb') as f:
            # Read and update hash in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"ERROR: Failed to calculate hash for {filepath}: {e}")
        return None # Return None if hashing fails

def _get_cache_path(avatar_path: str):
    """Determines the path for the lipsync cache file."""
    # Save cache file next to the avatar file with a .lipsync_cache.json extension
    return f"{avatar_path}.lipsync_cache.json"

def _load_lipsync_cache(avatar_path: str, config):
    """Attempts to load lipsync cache data."""
    cache_path = _get_cache_path(avatar_path)
    if not os.path.exists(cache_path):
        logger.debug(f"DEBUG: Lipsync cache file not found at {cache_path}")
        return None

    logger.debug(f"DEBUG: Lipsync cache file found. Attempting to load from {cache_path}")
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # --- Validate Cache ---
        # 1. Check avatar path match (basic check, hash is more reliable)
        if cache_data.get('avatar_path') != avatar_path:
             logger.warning("WARNING: Cache avatar path mismatch. Cache invalid.")
             return None

        # 2. Check avatar file hash
        current_avatar_hash = _calculate_file_hash(avatar_path)
        if current_avatar_hash is None or cache_data.get('avatar_hash') != current_avatar_hash:
            logger.warning("WARNING: Cache avatar file hash mismatch. Cache invalid.")
            return None

        # 3. Check relevant configuration values
        # Add checks for any config values that affect face detection or box processing
        if cache_data.get('config_used', {}).get('FACE_DET_BATCH_SIZE') != config.FACE_DET_BATCH_SIZE:
             logger.warning("WARNING: Cache config (FACE_DET_BATCH_SIZE) mismatch. Cache invalid.")
             return None

        # 4. Check for required keys in cache data
        required_keys = ['avatar_path', 'avatar_hash', 'bounding_boxes', 'is_static', 'avatar_fps', 'avatar_duration', 'config_used']
        if not all(key in cache_data for key in required_keys):
             logger.warning("WARNING: Cache data is missing required keys. Cache invalid.")
             return None

        logger.debug("DEBUG: Lipsync cache validated successfully.")
        # Return the loaded data (raw bounding boxes, avatar info)
        return cache_data

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"ERROR: Failed to load or validate lipsync cache from {cache_path}: {e}")
        # Clean up potentially corrupted cache file
        try:
             os.remove(cache_path)
             logger.debug(f"DEBUG: Removed corrupted cache file: {cache_path}")
        except: pass # Ignore cleanup errors
        return None # Return None if loading or validation fails

def _save_lipsync_cache(avatar_path: str, raw_bounding_boxes: list, is_static: bool, avatar_fps: float, avatar_duration, config):
    """Saves lipsync cache data to a file."""
    cache_path = _get_cache_path(avatar_path)
    logger.debug(f"DEBUG: Attempting to save lipsync cache to {cache_path}")

    avatar_hash = _calculate_file_hash(avatar_path)
    if avatar_hash is None:
         logger.warning(f"WARNING: Could not calculate avatar hash for saving cache.")
         return # Don't save if cannot hash

    # Gather relevant config values to save
    config_to_save = {
         'FACE_DET_BATCH_SIZE': config.FACE_DET_BATCH_SIZE
    }

    cache_data = {
        'avatar_path': avatar_path,
        'avatar_hash': avatar_hash,
        'bounding_boxes': raw_bounding_boxes,
        'is_static': is_static,
        'avatar_fps': avatar_fps,
        'avatar_duration': avatar_duration,
        'config_used': config_to_save,
        'timestamp': time.time()
    }

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=4)
        logger.debug(f"DEBUG: Lipsync cache successfully saved to {cache_path}")
    except Exception as e:
        logger.error(f"ERROR: Failed to save lipsync cache to {cache_path}: {e}")


# --- Main Initialization Function ---

def initialize_lipsync(app_state: object):
    """
    Initializes the Lipsync service by loading models, the enhancer,
    preparing the avatar (using cache if available), then loads the
    OVWav2Lip pipeline into the application state.
    """
    logger.info("INFO: Initializing Lipsync service...")
    start_time = time.time()

    logger.debug("DEBUG: Ensuring Lipsync directories exist...")
    os.makedirs(config.LIPSYNC_MODELS_DIR_FULL, exist_ok=True)
    os.makedirs(config.REALESRGAN_MODELS_DIR_FULL, exist_ok=True)
    os.makedirs(config.LIPSYNC_RESULTS_DIR_FULL, exist_ok=True)
    logger.debug("DEBUG: Ensured Lipsync directories exist.")

    # --- Initialize OpenVINO Core ---
    logger.debug("DEBUG: Initializing OpenVINO Core...")
    try:
        core = ov.Core()
        logger.debug("DEBUG: OpenVINO Core initialized.")
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize OpenVINO Core: {e}")
        raise RuntimeError(f"Lipsync Init Failed: OpenVINO Core error: {e}") from e

    ov_config = {ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY}


    # --- Load and Compile OpenVINO Models ---
    logger.debug("DEBUG: Loading and compiling OpenVINO models...")
    face_detection_model_xml = os.path.join(config.LIPSYNC_MODELS_DIR_FULL, config.FACE_DET_MODEL_XML)
    wav2lip_model_xml = os.path.join(config.LIPSYNC_MODELS_DIR_FULL, config.WAV2LIP_MODEL_XML)

    if not os.path.exists(face_detection_model_xml) or not os.path.exists(wav2lip_model_xml):
         raise RuntimeError(f"Lipsync Init Failed: Missing OpenVINO model files. Expected: {face_detection_model_xml}, {wav2lip_model_xml}")

    try:
        compiled_face_detector = core.compile_model(face_detection_model_xml, config.LIPSYNC_DEVICE, ov_config)
        logger.debug("DEBUG: Face Detector model compiled.")
        wav2_lip_model = core.read_model(model=wav2lip_model_xml)
        compiled_wav2lip_model = core.compile_model(model=wav2_lip_model, device_name=config.LIPSYNC_DEVICE, config=ov_config)
        logger.debug("DEBUG: Wav2Lip model compiled.")
    except Exception as e:
        logger.error(f"ERROR: Failed to compile OpenVINO models: {e}")
        raise RuntimeError(f"Lipsync Init Failed: Model compilation error: {e}") from e


    # --- Initialize RealESRGan Enhancer ---
    logger.debug(f"DEBUG: Initializing RealESRGan Enhancer on device {config.ENHANCER_DEVICE}...")
    try:
        enhancer = initialize_enhancer(model_name=config.REALESRGAN_MODEL_NAME, device=config.ENHANCER_DEVICE)
        logger.debug("DEBUG: RealESRGan Enhancer initialized.")
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize RealESRGan Enhancer: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise RuntimeError(f"Lipsync Init Failed: Enhancer initialization error: {e}") from e


    # --- Prepare Avatar (Load frames and perform initial face detection or load from cache) ---
    avatar_path = config.LIPSYNC_AVATAR_PATH
    logger.info(f"INFO: Preparing avatar from {avatar_path}...")

    if not os.path.exists(avatar_path):
         raise RuntimeError(f"Lipsync Init Failed: Avatar file not found at {avatar_path}")

    # --- Always load full frames ---
    full_frames = []
    is_static = False
    avatar_fps = hp.fps
    avatar_duration = None

    logger.debug("DEBUG: Loading avatar frames...")
    if avatar_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
        logger.debug("DEBUG: Loading static image.")
        frame = cv2.imread(avatar_path)
        if frame is None:
             raise RuntimeError(f"Lipsync Init Failed: Could not read image file: {avatar_path}")
        full_frames = [frame]
        is_static = True
        avatar_fps = hp.fps
        avatar_duration = 0
    else:
        logger.debug("DEBUG: Loading video.")
        try:
            video_stream = cv2.VideoCapture(avatar_path)
            if not video_stream.isOpened():
                raise IOError(f"Could not open video file {avatar_path}")

            avatar_fps = video_stream.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            avatar_duration = round(frame_count / avatar_fps, 2) if avatar_fps > 0 else 0

            frames = []
            logger.debug(f"DEBUG: Reading {frame_count} frames from video {avatar_path}...")
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    break
                frames.append(frame)
            video_stream.release()
            full_frames = frames

            if not full_frames:
                 raise RuntimeError(f"Lipsync Init Failed: No frames read from video {avatar_path}")

            is_static = False

            logger.debug(f"DEBUG: Finished reading {len(full_frames)} frames.")

        except Exception as e:
            logger.error(f"ERROR: Failed to load avatar video frames: {e}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise RuntimeError(f"Lipsync Init Failed: Video loading error: {e}") from e

    if not full_frames:
         raise RuntimeError(f"Lipsync Init Failed: No avatar frames loaded from {avatar_path}")

    logger.debug("DEBUG: Avatar frames loaded.")

    # --- Attempt to load face detection results from cache ---
    cached_data = _load_lipsync_cache(avatar_path, config)
    face_det_results = []

    if cached_data is not None:
        logger.info("INFO: Using cached face detection results.") # Changed to INFO
        # Reconstruct face_det_results from cached raw bounding boxes
        raw_bounding_boxes = cached_data['bounding_boxes']
        # Ensure consistent dimensions if processing was batched/varied before
        if len(raw_bounding_boxes) != len(full_frames):
             logger.warning("WARNING: Cache frame count mismatch with avatar frames. Re-running detection.")
             cached_data = None # Invalidate cache if counts don't match
        else:
            logger.debug("DEBUG: Reconstructing face_det_results from cache...")
            pads = [0, 15, -10, -10]

            for i, rect_raw in enumerate(raw_bounding_boxes):
                 # Assuming rect_raw is (x1, y1, x2, y2)
                 x1_raw, y1_raw, x2_raw, y2_raw = rect_raw[:4]

                 # Apply padding using current config/hardcoded pads
                 y1 = max(0, int(y1_raw) - pads[0])
                 y2 = min(full_frames[i].shape[0], int(y2_raw) + pads[1])
                 x1 = max(0, int(x1_raw) - pads[2])
                 x2 = min(full_frames[i].shape[1], int(x2_raw) + pads[3])

                 # Ensure valid coordinates after padding and clipping
                 if x1 >= x2 or y1 >= y2:
                      logger.error(f"ERROR: Invalid reconstructed face bounding box for frame {i}: ({x1},{y1},{x2},{y2}). Skipping frame or handling error.")
                      raise RuntimeError(f"Lipsync Init Failed: Invalid reconstructed face box for frame {i} from cache.")

                 cropped_img = full_frames[i][y1: y2, x1:x2]
                 face_det_results.append([cropped_img, (y1, y2, x1, x2)]) # Store [cropped_img, bbox_tuple]

            logger.debug("DEBUG: Face detection results reconstructed from cache.")

    # --- Perform face detection if cache is not used or invalid ---
    if cached_data is None:
        logger.info("INFO: Performing face detection on avatar frames...") # Changed to INFO

        try:
            face_detector_wrapper = OVFaceAlignment(
                 LandmarksType._2D, face_detector=compiled_face_detector, flip_input=False, device=config.LIPSYNC_DEVICE) # Pass compiled model

            initial_batch_size = config.FACE_DET_BATCH_SIZE
            current_batch_size = initial_batch_size
            all_predictions_raw = []

            # Loop to potentially reduce batch size if memory error occurs
            for attempt in range(int(np.log2(initial_batch_size)) + 2):
                try:
                    all_predictions_raw = [] # Clear predictions for retry
                    logger.debug(f"DEBUG: Attempting face detection batching with batch size: {current_batch_size}")
                    for i in range(0, len(full_frames), current_batch_size):
                        batch_images = np.array(full_frames[i:i + current_batch_size])
                        # Ensure images are in BGR if detector expects (cv2 reads BGR)
                        batch_predictions = face_detector_wrapper.get_detections_for_batch(batch_images)
                        all_predictions_raw.extend(batch_predictions)
                    logger.debug("DEBUG: Face detection batching complete.")
                    break # Exit retry loop on success
                except RuntimeError as e:
                    if current_batch_size > 1:
                        logger.warning(f"WARNING: Face detection batching failed with size {current_batch_size}: {e}. Reducing batch size.")
                        current_batch_size //= 2
                        if all_predictions_raw:
                             logger.warning("WARNING: Some predictions gathered before error. Discarding partial results.")
                             all_predictions_raw = []
                    else:
                        logger.error(f"ERROR: Face detection failed even with batch size 1: {e}")
                        raise # Re-raise if batch size is already 1 or reduction fails

            if not all_predictions_raw and len(full_frames) > 0:
                 raise RuntimeError("Lipsync Init Failed: Face detection failed for all frames after retries.")

            # Process raw predictions into face_det_results list [cropped_img, (y1, y2, x1, x2)]
            face_det_results = []
            pads = [0, 15, -10, -10]

            for i, rect_raw in enumerate(all_predictions_raw):
                if rect_raw is None:
                    logger.error(f"ERROR: Face not detected in frame {i} of avatar: {avatar_path}")
                    raise RuntimeError(f"Lipsync Init Failed: Face not detected in frame {i} of avatar.")

                # Assuming rect_raw format is (x1, y1, x2, y2)
                x1_raw, y1_raw, x2_raw, y2_raw = rect_raw[:4]

                # Apply padding
                y1 = max(0, int(y1_raw) - pads[0])
                y2 = min(full_frames[i].shape[0], int(y2_raw) + pads[1])
                x1 = max(0, int(x1_raw) - pads[2])
                x2 = min(full_frames[i].shape[1], int(x2_raw) + pads[3])

                if x1 >= x2 or y1 >= y2:
                     logger.error(f"ERROR: Invalid face bounding box after padding in frame {i}: ({x1},{y1},{x2},{y2})")
                     raise RuntimeError(f"Lipsync Init Failed: Invalid face box in frame {i} of avatar.")

                cropped_img = full_frames[i][y1: y2, x1:x2]
                face_det_results.append([cropped_img, (y1, y2, x1, x2)])

            logger.debug("DEBUG: Face detection on avatar frames complete.")

            # --- Save face detection results to cache ---
            logger.debug("DEBUG: Saving face detection results to cache...")
            # Save the RAW predictions before padding/smoothing
            _save_lipsync_cache(avatar_path, all_predictions_raw, is_static, avatar_fps, avatar_duration, config)
            logger.debug("DEBUG: Face detection results saved to cache.")

        except Exception as e:
             logger.error(f"ERROR: Failed face detection during avatar processing: {e}")
             traceback.print_exc(file=sys.stdout)
             sys.stdout.flush()
             raise RuntimeError(f"Lipsync Init Failed: Face detection error: {e}") from e


    # --- Initialize OVWav2Lip Pipeline ---
    logger.debug("DEBUG: Initializing OVWav2Lip pipeline object...")
    try:
        wav2lip_pipeline = OVWav2Lip(
            compiled_face_detector=compiled_face_detector,
            compiled_wav2lip_model=compiled_wav2lip_model,
            enhancer=enhancer,
            full_frames=full_frames,
            face_det_results=face_det_results,
            fps=avatar_fps,
            duration=avatar_duration,
            static=is_static
        )
        logger.debug("DEBUG: OVWav2Lip pipeline object initialized.")
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize OVWav2Lip pipeline: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise RuntimeError(f"Lipsync Init Failed: OVWav2Lip object error: {e}") from e


    # --- Perform Warming Up ---
    logger.info("INFO: Starting Lipsync warming up inference...")
    warmup_start_time = time.time()
    try:
        temp_filename = "lipsync_warmup_dummy.wav"
        warmup_audio_path = os.path.join(config.DATA_DIRECTORY, temp_filename)
        if not os.path.exists(warmup_audio_path):
            with wave.open(warmup_audio_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                num_samples = 16000 * 2 # 2 seconds
                wf.writeframes(np.zeros(num_samples, dtype=np.int16).tobytes())
            logger.debug(f"DEBUG: Created dummy warmup audio at {warmup_audio_path}.")
        else:
             logger.debug(f"DEBUG: Using existing dummy warmup audio at {warmup_audio_path}.")


        # Perform warming up inference
        warmup_result_file_id = wav2lip_pipeline.inference(warmup_audio_path, enhance=False, output_dir=config.LIPSYNC_RESULTS_DIR_FULL)
        warmup_end_time = time.time()
        logger.info(f"INFO: Warming up inference complete in {warmup_end_time - warmup_start_time:.2f} seconds. Result ID: {warmup_result_file_id}") # Log time and ID

        # Clean up dummy audio file and generated video result
        try:
             if os.path.exists(warmup_audio_path):
                  os.remove(warmup_audio_path)
             warmup_result_path = os.path.join(config.LIPSYNC_RESULTS_DIR_FULL, f"{warmup_result_file_id}.mp4")
             if os.path.exists(warmup_result_path): # Check existence before removing
                  os.remove(warmup_result_path)
             logger.debug("DEBUG: Warming up cleanup complete.")
        except Exception as cleanup_e:
             logger.warning(f"WARNING: Failed to clean up warmup files: {cleanup_e}")


    except Exception as e:
         logger.error(f"ERROR: Failed during Lipsync warming up inference: {e}")
         traceback.print_exc(file=sys.stdout)
         sys.stdout.flush()
         # Decide if warm-up failure should crash startup. Logging error might be enough.
         # raise RuntimeError(f"Lipsync Init Failed: Warming up error: {e}") from e


    # --- Store Pipeline in App State ---
    app_state.wav2lip_pipeline = wav2lip_pipeline
    end_time = time.time() # End timing here
    logger.info(f"INFO: Lipsync service initialization complete in {end_time - start_time:.2f} seconds.") # Log total time

