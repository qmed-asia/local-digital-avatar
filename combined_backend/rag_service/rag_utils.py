
import os
import sys
import logging

import combined_backend.config as config

from .utils.chroma_client import ChromaClient

logger = logging.getLogger("uvicorn.error")

def initialize_rag(app_state: object):
    """
    Initializes the RAG service by setting up ChromaDB, loading embeddings,
    and reranker models, then loads the ChromaClient into the application state.
    """
    logger.info("DEBUG: Initializing RAG service (from rag_utils)...")

    # Ensure necessary directories exist (vector db and doc store)
    logger.debug("DEBUG: Ensuring RAG directories exist...")
    os.makedirs(config.RAG_VECTORDB_DIR, exist_ok=True)
    os.makedirs(config.RAG_DOCSTORE_DIR, exist_ok=True)
    # Note: ChromaClient will handle creating the collection itself
    logger.debug(f"DEBUG: RAG directories ensured: {config.RAG_VECTORDB_DIR}, {config.RAG_DOCSTORE_DIR}")

    # --- Initialize ChromaClient ---
    # The ChromaClient's __init__ method handles loading/downloading/compiling
    # embedding and reranker models using OpenVINO.
    logger.debug(f"DEBUG: Initializing ChromaClient with db_dir={config.RAG_VECTORDB_DIR}, embedding_device={config.RAG_EMBEDDING_DEVICE}, reranker_device={config.RAG_RERANKER_DEVICE}...")

    try:
        chroma_client_instance = ChromaClient(
            db_dir=config.RAG_VECTORDB_DIR,
            embedding_device=config.RAG_EMBEDDING_DEVICE,
            reranker_device=config.RAG_RERANKER_DEVICE
        )
        logger.debug("DEBUG: ChromaClient initialized successfully.")

    except Exception as e:
        logger.error(f"ERROR: Failed to initialize ChromaClient: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise RuntimeError(f"RAG Init Failed: ChromaClient error: {e}") from e

    # --- Store ChromaClient in App State ---
    app_state.rag_chroma_client = chroma_client_instance
    logger.info("DEBUG: RAG service initialization complete (from rag_utils).")
