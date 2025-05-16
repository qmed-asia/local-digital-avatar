
import os
import io
import sys
import time
import requests
import json
import numpy as np
from typing import List, Union, Optional
from typing_extensions import TypedDict, Required 

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel 
from fastapi.encoders import jsonable_encoder 

import combined_backend.config as config

from ..rag_service.utils.chroma_client import ChromaClient
from ..rag_service.utils.prompt import RAG_PROMPT, NO_CONTEXT_FOUND_PROMPT, QUERY_REWRITE_PROMPT
from openai import OpenAI


llm_client = OpenAI(
    api_key=config.LLM_API_KEY, 
    base_url=config.LLM_BASE_URL
)


router = APIRouter(
    prefix="/rag", 
    tags=["RAG"],
)

# --- Define Request/Response Models ---
class Message(BaseModel):
    role: str = "user"
    content: str

class ICreateChatCompletions(BaseModel): # Use BaseModel for request body validation
    messages: List[Message]
    model: str = "qwen3:8b"
    tools: Optional[List] = None # Optional
    rag: bool = False # Default False
    endpoint: Optional[str] = None # Optional
    suffix: Optional[str] = None # Optional
    max_tokens: int = 50 # Default 16
    temperature: Union[int, float] = 1 # Default 1
    top_p: Union[int, float] = 1 # Default 1
    n: int = 1 # Default 1
    stream: bool = False # Default False
    logprobs: Optional[int] = None # Optional
    echo: bool = False # Default False
    stop: Optional[Union[str, List[str]]] = None # Optional
    presence_penalty: float = 0 # Default 0
    frequency_penalty: float = 0 # Default 0
    best_of: Optional[int] = None # Optional
    logit_bias: Optional[dict[str, float]] = None # Optional
    user: Optional[str] = None # Optional
    top_k: int = -1 # Default -1
    ignore_eos: bool = False # Default False
    use_beam_search: bool = False # Default False
    stop_token_ids: Optional[List[int]] = None # Optional
    skip_special_tokens: bool = True # Default True

class IModel(BaseModel):
    model: str = "qwen3:8b"





# --- RAG Data Management Endpoints ---
@router.get("/text_embeddings", status_code=200)
async def get_text_embeddings(request: Request, page: Optional[int] = 1, pageSize: Optional[int] = 5, source: Optional[str] = ""):
    """Retrieves paginated text chunks and metadata from the vector database."""
    chroma_client: ChromaClient = request.app.state.rag_chroma_client
    if chroma_client is None: raise HTTPException(status_code=503, detail="RAG service not initialized")

    data = chroma_client.get_all_collection_data(page, pageSize, source)
    result = {"status": True, "data": data}
    return JSONResponse(content=jsonable_encoder(result))

@router.get("/text_embedding_sources", status_code=200)
async def get_text_embedding_sources(request: Request):
    """Lists unique document sources in the vector database."""
    chroma_client: ChromaClient = request.app.state.rag_chroma_client
    if chroma_client is None: raise HTTPException(status_code=503, detail="RAG service not initialized")

    data = chroma_client.get_all_sources()
    result = {"status": True, "data": data}
    return JSONResponse(content=jsonable_encoder(result))

@router.delete("/text_embeddings/{uuid}", status_code=200)
async def delete_text_embeddings_by_uuid(request: Request, uuid: str):
    """Deletes a specific embedding by its UUID."""
    chroma_client: ChromaClient = request.app.state.rag_chroma_client
    if chroma_client is None: raise HTTPException(status_code=503, detail="RAG service not initialized")

    data = chroma_client.delete_data(uuid)
    result = {"status": True, "data": data}
    return JSONResponse(content=jsonable_encoder(result))


@router.delete("/text_embeddings/source/{source}", status_code=200)
async def delete_text_embeddings_by_source(request: Request, source: str): 
    """Deletes all embeddings associated with a specific source file and potentially the source file itself."""
    chroma_client: ChromaClient = request.app.state.rag_chroma_client
    if chroma_client is None: raise HTTPException(status_code=503, detail="RAG service not initialized")

    # Construct the full source file path relative to the combined backend's DATA_DIRECTORY
    source_file_path = os.path.join(config.RAG_DOCSTORE_DIR, source)

    try:
        if os.path.isfile(source_file_path):
            # print(f"DEBUG: Removing the source file: {source_file_path}", flush=True)
            os.remove(source_file_path) 
            # print("DEBUG: Source file removed.", flush=True) 

        # Delete embeddings associated with the source
        data = chroma_client.delete_data_by_source(source)
        result = {"status": True, "data": data}
        # print(f"DEBUG: Embeddings for source {source} deleted.", flush=True)

    except FileNotFoundError:
         print(f"WARNING: Attempted to delete source file {source_file_path} but not found.", flush=True)
         data = chroma_client.delete_data_by_source(source)
         result = {"status": True, "data": data, "detail": f"Source file {source} not found, but embeddings deleted."}
    except Exception as error:
        print(f"ERROR: Failed to delete data source {source}: {error}", flush=True)
        result = {"status": False, "data": None, "detail": str(error)} 

    return JSONResponse(content=jsonable_encoder(result))


@router.post("/text_embeddings", status_code=200)
async def upload_rag_doc(request: Request, chunk_size: int = Form(...), chunk_overlap: int = Form(...), files: List[UploadFile] = File(...)): # Use Form for mixed data
    """
    Uploads document files (.pdf, .txt), processes them, and creates/saves embeddings.
    Accepts multipart/form-data with chunk_size, chunk_overlap (Form fields) and files (File fields).
    """
    print(f"DEBUG: Received /rag/v1/text_embeddings upload request with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, num_files={len(files)}", flush=True)

    chroma_client: ChromaClient = request.app.state.rag_chroma_client
    if chroma_client is None: raise HTTPException(status_code=503, detail="RAG service not initialized")

    ALLOWED_EXTENSIONS = ['.pdf', '.txt']
    file_list = []
    processed_filenames = [] 

    if not os.path.isdir(config.RAG_DOCSTORE_DIR):
        os.makedirs(config.RAG_DOCSTORE_DIR, exist_ok=True)
        print(f"DEBUG: Created document store directory: {config.RAG_DOCSTORE_DIR}", flush=True)

    for file in files:
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            print(f"WARNING: Skipping unsupported file type: {file.filename}", flush=True)
            continue 

        # Save the uploaded file to the document store directory
        file_path = os.path.join(config.RAG_DOCSTORE_DIR, file.filename)
        print(f"DEBUG: Saving uploaded file to: {file_path}", flush=True)
        try:
            # Read file content in chunks to avoid large memory usage
            with open(file_path, "wb") as f:
                f.write(file.file.read()) 
            print(f"DEBUG: File saved: {file.filename}", flush=True)
            processed_filenames.append(file.filename)

        except Exception as e:
            print(f"ERROR: Failed to save uploaded file {file.filename}: {e}", flush=True)
            continue

    if len(processed_filenames) == 0:
        print("ERROR: No valid files were uploaded for processing.", flush=True) 
        raise HTTPException(status_code=400, detail="No supported files were uploaded for creating text embeddings.")

    print(f"DEBUG: Creating collection data for processed files: {processed_filenames}", flush=True) 
    try:
        success = chroma_client.create_collection_data(
            processed_filenames, chunk_size, chunk_overlap
        )
        print(f"DEBUG: create_collection_data finished. Success: {success}", flush=True) 

        if not success: # Check the return status from chroma_client
             print("ERROR: ChromaClient.create_collection_data reported failure.", flush=True) 
             raise RuntimeError("Failed to create collection data in ChromaDB.")

        result = {"status": True, "data": processed_filenames}
        print("DEBUG: Upload and embedding process completed successfully.", flush=True) 
        return JSONResponse(content=jsonable_encoder(result))

    except Exception as error:
        print(f"ERROR: An error occurred during embedding creation: {error}", flush=True) 
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        # Decide if partial uploads/saves should be cleaned up on error
        raise HTTPException(status_code=500, detail=f"Failed to create text embeddings: {error}")


# --- LLM / Chat Completions Endpoints ---
@router.get("/models", status_code=200)
async def get_available_model():
    """Lists available LLM models from the configured LLM service (Ollama)."""
    print("DEBUG: Received /rag/v1/models request.", flush=True) 
    try:
        model_list = llm_client.models.list()
        print("DEBUG: Successfully retrieved model list from LLM service.", flush=True) 
        return JSONResponse(content=jsonable_encoder(model_list))
    except Exception as e:
         print(f"ERROR: Failed to get model list from LLM service: {e}", flush=True) 
         # Check if it's a connection error to the LLM service
         if isinstance(e, requests.exceptions.ConnectionError):
              raise HTTPException(status_code=502, detail=f"Failed to connect to LLM service at {config.LLM_BASE_URL}. Please ensure Ollama is running and accessible.")
         elif hasattr(e, '_message') and "Connection error" in e._message:
             raise HTTPException(status_code=502, detail=f"Failed to connect to LLM service at {config.LLM_BASE_URL}. Please ensure Ollama is running and accessible.")
         else:
             raise HTTPException(status_code=500, detail=f"Failed to retrieve model list: {e}")


@router.post("/pull", status_code=200)
async def pull_model(data: IModel):
    """
    Proxies a model pull request to the configured LLM service (Ollama).
    Note: Ollama's pull endpoint often streams, but original RAG backend
    commented out the streaming part. We'll stick to non-streaming proxy for now.
    """
    model = data.model
    print(f"DEBUG: Received /rag/v1/pull request for model: {model}", flush=True) 


    llm_base_url_no_v1 = config.LLM_BASE_URL.replace("/v1", "") if config.LLM_BASE_URL.endswith("/v1") else config.LLM_BASE_URL
    pull_endpoint = f"{llm_base_url_no_v1}/api/pull"
    print(f"DEBUG: Proxying pull request to LLM service at: {pull_endpoint}", flush=True) 

    try:
        response = requests.post(
            pull_endpoint,
            json={"model": model, "stream": False}, 
            timeout=300 # 
        )
        response.raise_for_status() 

        print(f"DEBUG: LLM pull request finished. Status code: {response.status_code}", flush=True) 

        # Return the JSON response received from the LLM service
        return JSONResponse(content=jsonable_encoder({"status": True, "data": response.json()}))

    except requests.exceptions.ConnectionError:
         print(f"ERROR: Failed to connect to LLM service at {pull_endpoint} during pull.", flush=True) 
         raise HTTPException(status_code=502, detail=f"Failed to connect to LLM service at {pull_endpoint}. Please ensure Ollama is running and accessible.")
    except requests.exceptions.Timeout:
         print(f"ERROR: LLM pull request timed out after 300 seconds.", flush=True) 
         raise HTTPException(status_code=504, detail="LLM model pull request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error during LLM pull request: {e}", flush=True) 
        try:
            error_detail = response.json()
        except:
            error_detail = {"message": str(e)}
        raise HTTPException(status_code=response.status_code if response else 500, detail=f"Error proxying pull request: {error_detail.get('message', str(e))}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred during model pull: {e}", flush=True) 
         raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model pull: {e}")


@router.post("/chat/completions", status_code=200)
async def chat_completion(request: Request, data: ICreateChatCompletions):
    """
    Handles chat completions requests, optionally incorporating RAG if enabled.
    Proxies the final request to the configured LLM service (Ollama).
    """
    def _formatting_rag_fusion_result(retrieval_list, query):
        context = ""
        for i, item in enumerate(retrieval_list):
            document = item['document']
            score = np.array(item['score']) * 100
            context += f"Context {i+1}: {document}.\nScore: {score:.2}."
            if i < len(retrieval_list) - 1:
                context += "\n\n"

        formatted_prompt = RAG_PROMPT.format(
            context=context,
            question=query
        )
        return formatted_prompt
    
    # print("DEBUG: Received /rag/v1/chat/completions request.", flush=True) 
    # print(f"DEBUG: Request body data: {data.model_dump_json(indent=2)}", flush=True) # Verbose log


    chroma_client: ChromaClient = request.app.state.rag_chroma_client
    if chroma_client is None:
         print("ERROR: ChromaClient not initialized for chat completions.", flush=True) 
         raise HTTPException(status_code=503, detail="RAG service not initialized")


    isRAG_enabled_via_header = False # Default RAG off
    # Check for 'rag' header (case-insensitive)
    rag_header = request.headers.get("rag", "").upper()
    if rag_header == "ON":
        isRAG_enabled_via_header = True
        print("DEBUG: RAG enabled via header.", flush=True) 
    elif rag_header == "OFF":
        isRAG_enabled_via_header = False
        print("DEBUG: RAG disabled via header.", flush=True) 

    # Check if RAG is enabled by request body OR header
    isRAG_enabled = data.rag or isRAG_enabled_via_header
    print(f"DEBUG: Final RAG status (body={data.rag}, header={rag_header}): {isRAG_enabled}", flush=True) 


    if isRAG_enabled:
        print("DEBUG: RAG settings is enabled. Verifying embedding is available", flush=True) 
        try:
             num_embeddings = chroma_client.get_num_embeddings()
             print(f"DEBUG: Number of embeddings available: {num_embeddings}", flush=True) 
             isEmbeddings = num_embeddings > 0
        except Exception as e:
             print(f"ERROR: Failed to get number of embeddings: {e}", flush=True) 
             isEmbeddings = False 


        if isEmbeddings:
            # Find the last user message
            last_user_message = None
            for message in reversed(data.messages):
                 if message.role == 'user':
                      last_user_message = message.content
                      break # Found the last user message

            if last_user_message:
                print(f"DEBUG: Last user message: {last_user_message}", flush=True) 
                print("DEBUG: Setting the temperature to 0.01 for RAG query to LLM.", flush=True) 
                data.temperature = 0.01 # Modify BaseModel instance directly if mutable

                print("DEBUG: Querying vector database...", flush=True) 
                reranker_results = chroma_client.query_data(last_user_message) # Use stored client
                print(f"DEBUG: Vector DB query returned {len(reranker_results)} results.", flush=True) 

                if len(reranker_results) == 0:
                    print("WARNING: Failed to retrieve contexts for RAG. No results from vector DB.", flush=True) 

                    user_prompt_content = NO_CONTEXT_FOUND_PROMPT.format(question=last_user_message)
                    print("DEBUG: Using NO_CONTEXT_FOUND_PROMPT.", flush=True) 

                else:

                    rag_score_threshold = 0 # Default threshold if not configured

                    # Filter results (assuming reranker_results format matches)
                    filtered_results = [elem for elem in reranker_results if elem.get('score', 0) > rag_score_threshold]
                    print(f"DEBUG: Filtered vector DB results (score > {rag_score_threshold}): {len(filtered_results)}", flush=True) 

                    if len(filtered_results) > 0:
                        print("DEBUG: Successfully retrieved and filtered contexts for RAG.", flush=True) 
                        # Format the user prompt with context
                        user_prompt_content = _formatting_rag_fusion_result(
                            filtered_results, last_user_message)
                        print("DEBUG: Using RAG_PROMPT with context.", flush=True) 
                        # print(f"DEBUG: Formatted RAG Prompt:\n{user_prompt_content}", flush=True) # Verbose log
                    else:
                        print("WARNING: No results passed score threshold for RAG.", flush=True) 
                        user_prompt_content = NO_CONTEXT_FOUND_PROMPT.format(
                            question=last_user_message)
                        print("DEBUG: Using NO_CONTEXT_FOUND_PROMPT.", flush=True) 

                # Update the last user message in the data object with the RAG prompt
                found_last_user_message = False
                for message in reversed(data.messages):
                     if message.role == 'user':
                          message.content = user_prompt_content 
                          found_last_user_message = True
                          break

                if not found_last_user_message:
                     print("ERROR: Could not find last user message to update content for RAG.", flush=True) 
                     pass

            else:
                 print("WARNING: No last user message found in the request for RAG processing.", flush=True) 


        else:
            print("WARNING: RAG enabled, but no embeddings available in vector DB. Skipping RAG processing.", flush=True) 


    # --- Prepare and Proxy Request to LLM Service ---
    if data.endpoint is not None:
        llm_endpoint = data.endpoint
        print(f"DEBUG: Using endpoint provided in request body: {llm_endpoint}", flush=True) 
    else:
        # Use the configured LLM_BASE_URL and the standard chat completions path
        llm_endpoint = f"{config.LLM_BASE_URL}/chat/completions"
        print(f"DEBUG: Using default configured endpoint: {llm_endpoint}", flush=True) 

    llm_request_data = data.model_dump(exclude={"rag"})

    # Ensure 'endpoint' key is removed if it was only for our internal routing
    llm_request_data.pop('endpoint', None)


    print(f"DEBUG: Proxying chat completion request to LLM service at: {llm_endpoint}", flush=True) 
    # print(f"DEBUG: LLM request data: {json.dumps(llm_request_data, indent=2)}", flush=True) # Verbose log

    llm_response = None

    try:
        llm_response = requests.post(
            llm_endpoint,
            json=llm_request_data,
            stream=True if data.stream else False, 
            timeout=300
        )
        llm_response.raise_for_status()

        print(f"DEBUG: LLM chat completion request finished. Status code: {llm_response.status_code}", flush=True) 


        # Return StreamingResponse if client requested stream, otherwise handle non-stream
        if data.stream:
            print("DEBUG: Returning StreamingResponse for LLM stream.", flush=True) 
            return StreamingResponse(_streamer(llm_response), media_type="text/event-stream") # Pass the response object

        else:
            print("DEBUG: Returning JSONResponse for non-streaming LLM response.", flush=True) 
            # Return the JSON response received from the LLM service
            return JSONResponse(content=jsonable_encoder(llm_response.json()))


    except requests.exceptions.ConnectionError as e:
         print(f"ERROR: Failed to connect to LLM service at {llm_endpoint} during chat completion.", flush=True) 
         raise HTTPException(status_code=502, detail=f"Failed to connect to LLM service at {llm_endpoint}. Please ensure Ollama is running and accessible.")
    except requests.exceptions.Timeout as e:
         print(f"ERROR: LLM chat completion request timed out after 300 seconds.", flush=True) 
         raise HTTPException(status_code=504, detail="LLM chat completion request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error during LLM request to {llm_endpoint}: {e}", flush=True)
        error_message = str(e) 
        status_code_to_return = 500 

        if e.response is not None:
            try:
                # Try to extract error detail from the response body if it's JSON
                error_detail_from_response = e.response.json()
                error_message = error_detail_from_response.get('message', error_message) 
            except json.JSONDecodeError:
                 # Handle case where response body is not valid JSON
                 error_message = f"LLM service returned non-JSON error body: {e.response.text[:100]}..."
            except Exception as json_e:
                 # Catch other potential errors during JSON parsing
                 error_message = f"Error parsing LLM service error response: {json_e}. Original error: {str(e)}"
            status_code_to_return = e.response.status_code 

        # Raise HTTPException with the determined status code and message
        raise HTTPException(status_code=status_code_to_return, detail=f"Error communicating with LLM service: {error_message}")
    
    except Exception as e:
         # This is a general catch-all for any other unexpected errors
         print(f"ERROR: An unexpected error occurred during chat completion: {e}", flush=True)
         import traceback
         traceback.print_exc(file=sys.stdout)
         sys.stdout.flush()

         # Default error info
         error_detail_info = {"message": str(e)}
         status_code_to_return = 500 


         if llm_response is not None:
             try:
                 error_detail_from_response = llm_response.json()
                 error_detail_info['message'] = error_detail_from_response.get('message', error_detail_info['message'])
             except:
                  pass

             status_code_to_return = llm_response.status_code

         raise HTTPException(status_code=status_code_to_return, detail=f"An unexpected error occurred: {error_detail_info['message']}")


# --- Helper function for streaming LLM response ---
def _streamer(llm_response):
    """Generator function to yield chunks from a streaming LLM response."""
    try:
        # Iterate over the streaming response content
        for chunk in llm_response.iter_content(chunk_size=1024): # Use a reasonable chunk size
            if chunk: # Yield non-empty chunks
                yield chunk
    except Exception as e:
        print(f"ERROR: Error during LLM streaming response: {e}", flush=True) 
        pass
