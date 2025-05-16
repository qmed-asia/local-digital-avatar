
# --- Initial Imports ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import combined_backend.config as config

# Utils function for each service
from combined_backend.stt_service.stt_utils import download_default_model, load_model_pipeline
from combined_backend.tts_service.tts_utils import initialize_piper_tts
from combined_backend.lipsync_service.lipsync_utils import initialize_lipsync
from combined_backend.rag_service.rag_utils import initialize_rag


# Import router for different services
from combined_backend.routers.stt_api import router as stt_router
from combined_backend.routers.tts_api import router as tts_router
from combined_backend.routers.lipsync_api import router as lipsync_router
from combined_backend.routers.rag_api import router as rag_router
from combined_backend.routers.avatar_api import router as avatar_router
from combined_backend.routers.avatar_streaming_api import router as avatar_streaming_router

logger = logging.getLogger("uvicorn.error")


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan function to initialize all services at startup and clean up at shutdown.
    Initializes services and attaches them to app.state.
    Uses configuration from the config module.
    """
    print("DEBUG: Application startup: Initializing services...", flush=True)
    start_time = time.time()

    # --- Initialize Common Directories ---
    # Ensure the temporary audio directory exists (used by STT and potentially TTS)
    if not os.path.exists(config.TEMP_AUDIO_DIR):
        os.makedirs(config.TEMP_AUDIO_DIR, exist_ok=True)
        print(f"DEBUG: Created temporary audio directory: {config.TEMP_AUDIO_DIR}", flush=True)

    # Ensure the main data directory exists
    if not os.path.exists(config.DATA_DIRECTORY):
         os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
         print(f"DEBUG: Created data directory: {config.DATA_DIRECTORY}", flush=True)

    # Ensure the models base directory exists
    models_base_dir = os.path.join(config.DATA_DIRECTORY, "models")
    if not os.path.exists(models_base_dir):
        os.makedirs(models_base_dir, exist_ok=True)
        print(f"DEBUG: Created models base directory: {models_base_dir}", flush=True)

    # Ensure the Piper TTS data directory exists (if used for saving output)
    if not os.path.exists(config.PIPER_TTS_DATA_DIR_FULL):
         os.makedirs(config.PIPER_TTS_DATA_DIR_FULL, exist_ok=True)
         print(f"DEBUG: Created Piper TTS data directory: {config.PIPER_TTS_DATA_DIR_FULL}", flush=True)

    try:
        # -----------Initalize STT Service---------------------------
        if not os.path.exists(config.STT_MODEL_DIR_FULL):
            logger.info(f"STT Model not found at {config.STT_MODEL_DIR_FULL}. Downloading default model...")
            # Call the download function from stt_utils.py
            # Use config values
            download_default_model(config.STT_DEFAULT_MODEL_ID, config.STT_MODEL_DIR_FULL)

        app.state.asr_pipeline = load_model_pipeline(config.STT_MODEL_DIR_FULL, device=config.STT_DEVICE)
        logger.info("DEBUG: STT service initialized successfully.")
        #-------------------------------------------------------------

        # ------------Initialize TTS Service--------------------------
        initialize_piper_tts(app.state)
        logger.info("DEBUG: TTS service initialized.")
        #-------------------------------------------------------------#

        # ---------- Initialize Lipsync service------------------------
        initialize_lipsync(app.state) # <-- Add this call
        #-------------------------------------------------------------#

        # -----------Initialize RAG service---------------------------
        initialize_rag(app.state)
        logger.info("DEBUG: RAG service initialized.")
        #-------------------------------------------------------------#

    except RuntimeError as e:
        print(f"ERROR: Application startup failed during Piper TTS service initialization: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail="Piper TTS service failed to initialize")


    print("DEBUG: All services initialized. Application is ready to serve.", flush=True)

    end_time = time.time()

    print(f"DEBUG: Initialization took {end_time - start_time} seconds.", flush=True)
    yield

    # --- Application Shutdown ---
    print("DEBUG: Application shutdown: Cleaning up services...", flush=True)


    print("DEBUG: Application shutdown complete.", flush=True)


# --- Create the main FastAPI app instance ---
app = FastAPI(
    title=os.getenv("APP_TITLE", "Digital Avatar Backend"),
    version=os.getenv("APP_VERSION", "1.0"), 
    description=os.getenv("APP_DESCRIPTION", "Combined backend for Digital Avatar services."),
    lifespan=lifespan 
)

# --- Add Common Middlewares ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Include Routers ---
app.include_router(stt_router, prefix="/v1")
app.include_router(tts_router, prefix="/v1")
app.include_router(lipsync_router, prefix="/v1")
app.include_router(rag_router, prefix="/v1")
app.include_router(avatar_router, prefix="/v1")
app.include_router(avatar_streaming_router, prefix="/v1")


# --- Global Healthcheck Endpoint ---
@app.get("/healthcheck", status_code=200)
async def main_healthcheck(request: Request):
    """
    Main healthcheck endpoint for the combined backend.
    Checks if core services are initialized by looking at app.state.
    """
    # Check if STT pipeline is initialized by looking at app.state
    if not hasattr(request.app.state, 'asr_pipeline') or request.app.state.asr_pipeline is None:
        print("DEBUG: Healthcheck failed: STT pipeline not initialized.", flush=True)
        raise HTTPException(status_code=503, detail="STT service is not ready")
    
    # Check if Piper TTS voices are loaded (Add this check)
    loaded_tts_voices = getattr(request.app.state, 'piper_tts_voices', {})
    if not loaded_tts_voices:
        print("DEBUG: Healthcheck failed: Piper TTS voices not loaded.", flush=True)
        raise HTTPException(status_code=503, detail="Piper TTS service is not ready or no voices loaded.")
    
        # Check RAG status # <-- Add RAG Status Check
    if hasattr(request.app.state, "rag_chroma_client") and request.app.state.rag_chroma_client is None:
        print("DEBUG: Healthcheck failed: RAG service is not ready.", flush=True)
        raise HTTPException(status_code=503, detail="RAG service is not ready")  

    print("DEBUG: Healthcheck successful.", flush=True)
    return {"status": "OK"}


# --- Main execution block for running with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)