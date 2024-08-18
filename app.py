import io
import logging
import os
import threading
from typing import List, Dict, Optional


import pandas as pd
from pydantic import BaseModel
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings

from tagmatch.fuzzysearcher import FuzzyMatcher
from tagmatch.logging_config import setup_logging
from tagmatch.vec_db import Embedder, VecDB

if not os.path.exists("./data"):
    os.makedirs("./data")

setup_logging(file_path="./data/service.log")


class Settings(BaseSettings):
    model_name: str
    cache_dir: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    embeddings_dim: Optional[int] = None
    class Config:
        env_file = ".env"


# Model for the override request
class OverrideRequest(BaseModel):
    tag: str
    synonyms: List[str]

settings = Settings()

app = FastAPI()
logger = logging.getLogger("fastapi")
# In-memory storage for tags
app.names_storage: List[str] = []
app.manual_overrides: Dict[str, List[str]] = {}

# Placeholder for the semantic search components
embedder = Embedder(model_name=settings.model_name, cache_dir=settings.cache_dir, target_dim=settings.embeddings_dim)
vec_db = VecDB(host=settings.qdrant_host,
               port=settings.qdrant_port,
               collection=settings.qdrant_collection,
               vector_size=settings.embeddings_dim if settings.embeddings_dim else embedder.embedding_dim)
app.fuzzy_matcher = FuzzyMatcher([])

# Flag to track background task status
task_running = threading.Event()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")

    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        logger.info(f"Request Body: {body.decode('utf-8')}")

    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.post("/upload-csv/")
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if task_running.is_set():
        raise HTTPException(
            status_code=400, detail="A task is already running. Please try again later.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400,
                            detail="Invalid file format. Please upload a CSV file (needs to env with '.csv'.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), sep=None, header=0)

    if 'name' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'name' column.")

    names_storage = df['name'].dropna().unique().tolist()

    if len(names_storage) == 0:
        raise HTTPException(status_code=400, detail="No names found in the CSV file.")

    # Return error if the collection is already existing (and it's populated)
    if vec_db.collection_exists():
        raise HTTPException(
            status_code=400, detail="Collection already exists. Please delete the collection first.")
    else:
        vec_db._create_collection()

    # Set the flag to indicate that a task is running
    task_running.set()

    # Add background task to process the CSV file
    background_tasks.add_task(process_csv, names_storage)

    return {"message": "File accepted for processing. Names will be extracted and stored in the background."}


def process_csv(names_storage: List[str]):
    try:
        if embedder.target_dim != embedder.embedding_dim:
            embedder.fit_pca(app.names_storage + names_storage)
        # Store embedded vectors for semantic search
        for name in names_storage:
            vector = embedder.embed(name)
            vec_db.store(vector, {"name": name})

        app.names_storage = names_storage
        app.fuzzy_matcher = FuzzyMatcher(app.names_storage)
    finally:
        # Clear the flag to indicate that the task has completed
        task_running.clear()


@app.delete("/purge/")
async def delete_collection():
    vec_db.remove_collection()
    app.names_storage = []
    return {"message": "DB deleted successfully."}


@app.post("/overrides/")
async def add_override(override_request: OverrideRequest):
    tag = override_request.tag
    synonyms = override_request.synonyms
    if not tag or not synonyms:
        raise HTTPException(status_code=400, detail="Tag and synonyms are required.")
    # Store the synonyms in the in-memory dictionary
    app.manual_overrides[tag] = synonyms

    # Embed and store each synonym in the vector database
    for synonym in synonyms:
        synonym_vector = embedder.embed(synonym)
        vec_db.store(synonym_vector, {"name": synonym, "original_tag": tag})

    return {"message": f"Override for '{tag}' added successfully.", "overrides": app.manual_overrides}


@app.get("/overrides/")
async def get_overrides():
    return {"manual_overrides": app.manual_overrides}


@app.delete("/overrides/{tag}")
async def delete_override(tag: str):
    if tag in app.manual_overrides:
        # Remove the synonyms from the vector database
        for synonym in app.manual_overrides[tag]:
            vec_db.delete_by_payload({"name": synonym, "original_tag": tag})

        # Remove the override from the in-memory dictionary
        del app.manual_overrides[tag]
        return {"message": f"Override for '{tag}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"No override found for tag '{tag}'.")


@app.get("/search/")
async def search(query: str, k: int = 5):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    # Fuzzy search
    fuzzy_matches = app.fuzzy_matcher.get_top_k_matches(query, k)

    # Semantic search
    query_vector = embedder.embed(query)
    semantic_matches = vec_db.find_closest(query_vector, k)

    # Include matches for synonyms
    if query in app.manual_overrides:
        for synonym in app.manual_overrides[query]:
            synonym_vector = embedder.embed(synonym)
            semantic_matches.extend(vec_db.find_closest(synonym_vector, k))

    # Remove duplicate results (if any) and sort by score
    seen = set()
    final_matches = []
    for match in semantic_matches:
        if match.payload["name"] not in seen:
            seen.add(match.payload["name"])
            final_matches.append(match)

    # Sort the final results by score
    final_matches = sorted(final_matches, key=lambda x: x.score, reverse=True)[:k]

    # Formatting the response
    semantic_results = [{"name": match.payload["name"], "score": match.score}
                        for match in final_matches]
    typo_results = [{"name": match["matched"], "score": match["score"]} for match in fuzzy_matches]

    response = {"match": {"semantic": semantic_results, "typo": typo_results}}

    return JSONResponse(content=response)


@app.get("/task-status/")
async def task_status():
    if task_running.is_set():
        return {"status": "running", "processed_items": vec_db.get_item_count()}
    else:
        return {"status": "finished", "nb_items_stored": vec_db.get_item_count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
