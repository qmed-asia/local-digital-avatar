
import os
import json
from dotenv import load_dotenv
from typing import Dict

load_dotenv(override=True)

# --- Path Constants ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIRECTORY = os.path.join(PROJECT_ROOT, "wav2lip", "results") 
TEMP_AUDIO_DIR = os.path.join(PROJECT_ROOT, "combined_backend", "tmp_audio") 


# --- Configuration Values (loaded from .env or environment) ---
## STT Configuration
STT_DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "openai/whisper-medium")
STT_DEVICE = os.getenv("STT_DEVICE", "CPU")
STT_MODEL_DIR_FULL = os.path.join(DATA_DIRECTORY, "models", STT_DEFAULT_MODEL_ID.split("/")[-1])


## Piper TTS Configuration
PIPER_TTS_DOWNLOAD_URL = os.getenv("PIPER_TTS_DOWNLOAD_URL", "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0")
PIPER_TTS_MODELS_SUBDIR = "piper-models"
PIPER_TTS_MODELS_DIR_FULL = os.path.join(DATA_DIRECTORY, "models", PIPER_TTS_MODELS_SUBDIR)
PIPER_TTS_DATA_SUBDIR = "piper-data"
PIPER_TTS_DATA_DIR_FULL = os.path.join(DATA_DIRECTORY, PIPER_TTS_DATA_SUBDIR)

## Define speaker paths
PIPER_TTS_SPEAKER_MODEL_PATHS: Dict[str, str] = {
    "female": "en/en_US/hfc_female/medium/en_US-hfc_female-medium",
    "male": "en/en_US/hfc_male/medium/en_US-hfc_male-medium"
}
PIPER_TTS_DEFAULT_SPEAKER = os.getenv("PIPER_TTS_DEFAULT_SPEAKER", "male")
PIPER_TTS_DEFAULT_LENGTH_SCALE = float(os.getenv("PIPER_TTS_DEFAULT_LENGTH_SCALE", "1.0"))


# --- Lipsync Service Configuration ---
# Device settings
LIPSYNC_DEVICE = os.getenv("LIPSYNC_DEVICE", "CPU")
ENHANCER_DEVICE = os.getenv("ENHANCER_DEVICE", "cpu")

# Subdirectories for Lipsync-related data within DATA_DIRECTORY/models
LIPSYNC_MODELS_SUBDIR = "lipsync-models"
REALESRGAN_MODELS_SUBDIR = "realesrgan-models" 

# Full paths for Lipsync models
LIPSYNC_MODELS_DIR_FULL = os.path.join(DATA_DIRECTORY, "models", LIPSYNC_MODELS_SUBDIR)
REALESRGAN_MODELS_DIR_FULL = os.path.join(DATA_DIRECTORY, "models", REALESRGAN_MODELS_SUBDIR)

# Specific OpenVINO model filenames (relative to LIPSYNC_MODELS_DIR_FULL)
FACE_DET_MODEL_XML = "face_detection.xml"
FACE_DET_MODEL_BIN = "face_detection.bin"
WAV2LIP_MODEL_XML = "wav2lip_gan.xml" 
WAV2LIP_MODEL_BIN = "wav2lip_gan.bin" 

# RealESRGan model name used in the initialize function
REALESRGAN_MODEL_NAME = os.getenv("REALESRGAN_MODEL_NAME", "realesr-animevideov3") 

# Subdirectory for Lipsync results within DATA_DIRECTORY
LIPSYNC_RESULTS_SUBDIR = "lipsync-results"
LIPSYNC_RESULTS_DIR_FULL = os.path.join(DATA_DIRECTORY, LIPSYNC_RESULTS_SUBDIR)

# Default avatar video/image path (relative to PROJECT_ROOT)
LIPSYNC_AVATAR_PATH = os.getenv("LIPSYNC_AVATAR_PATH", os.path.join(PROJECT_ROOT, "assets/video.mp4"))

# Operational settings
FACE_DET_BATCH_SIZE = int(os.getenv("FACE_DET_BATCH_SIZE", "8"))
PADS = [int(x) for x in os.getenv("LIPSYNC_PADS", "0,15,-10,-10").split(',')] 
NOSMOOTH_BOXES = os.getenv("NOSMOOTH_BOXES", "True").lower() == "true"


# --- End Lipsync Service Configuration ---

# --- RAG Service Configuration ---
RAG_VECTORDB_DIR = os.path.join(DATA_DIRECTORY, "embeddings") # Used by ChromaClient
RAG_DOCSTORE_DIR = os.path.join(RAG_VECTORDB_DIR, "documents") # Where source documents are stored

RAG_EMBEDDING_MODEL_LOCAL_DIR = os.path.join(DATA_DIRECTORY, "models", "embeddings", "bge-large-en-v1.5") # Used by ChromaClient
RAG_RERANKER_MODEL_LOCAL_DIR = os.path.join(DATA_DIRECTORY, "models", "reranker", "bge-reranker-large") # Used by ChromaClient


# Devices (Used by ChromaClient for OpenVINO models)
RAG_EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "CPU")
RAG_RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "CPU")

# LLM Service Configuration
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")


# RAG Parameters (from ChromaClient defaults or potential env vars)
RAG_VECTOR_SEARCH_TOP_K = int(os.getenv("RAG_VECTOR_SEARCH_TOP_K", "4")) # Used by ChromaClient
RAG_VECTOR_RERANK_TOP_N = int(os.getenv("RAG_VECTOR_RERANK_TOP_N", "3")) # Used by ChromaClient

# --- End RAG Service Configuration ---

# Add other common configuration variables here
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
ALLOWED_CORS_ORIGINS = json.loads(os.getenv("ALLOWED_CORS", '["*"]'))


# --- Background Task Settings ---
BACKGROUND_CLEANUP_DELAY = 30