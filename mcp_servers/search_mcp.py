import os
import logging
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# Assuming schemas.py is accessible.
try:
    from schemas import (
        DocumentMetadata, ListDocumentsResponse,
        DocumentContentResponse, SearchRequest,
        SearchResultItem, SearchResponse
    )
except ImportError:
    print("Warning: schemas.py not found in default PYTHONPATH. Redefining local schemas for Search MCP.")
    class DocumentMetadata(BaseModel):
        id: str
        filename: str
        content_preview: str

    class ListDocumentsResponse(BaseModel):
        success: bool = True
        documents: Optional[List[DocumentMetadata]] = None
        error: Optional[str] = None

    class DocumentContentResponse(BaseModel):
        success: bool = True
        id: Optional[str] = None
        filename: Optional[str] = None
        content: Optional[str] = None
        error: Optional[str] = None

    class SearchRequest(BaseModel):
        query: str

    class SearchResultItem(BaseModel):
        id: str
        filename: str
        score: float
        content_preview: str

    class SearchResponse(BaseModel):
        success: bool = True
        results: Optional[List[SearchResultItem]] = None
        error: Optional[str] = None

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Document storage path
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'documents')
logger.info(f"Search MCP documents directory configured to: {DOCUMENTS_DIR}")

# In-memory store for document metadata and content
# This would be loaded from disk in a real application
DOCUMENTS_DB: Dict[str, Dict[str, Any]] = {}

def load_documents_from_disk():
    """Loads documents from the DOCUMENTS_DIR into DOCUMENTS_DB."""
    global DOCUMENTS_DB
    if DOCUMENTS_DB: # Avoid reloading if already loaded, unless forced
        # For a real app, might need a refresh mechanism
        # logger.info("Documents already loaded.")
        return

    if not os.path.exists(DOCUMENTS_DIR):
        logger.warning(f"Documents directory {DOCUMENTS_DIR} not found. No documents will be loaded.")
        DOCUMENTS_DB = {} # Ensure it's empty
        return

    loaded_docs = {}
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"): # Consider other types if needed
            doc_id = os.path.splitext(filename)[0]
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                loaded_docs[doc_id] = {
                    "id": doc_id,
                    "filename": filename,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "full_content": content
                }
            except Exception as e:
                logger.error(f"Error loading document {filename}: {e}")
    DOCUMENTS_DB = loaded_docs
    logger.info(f"Loaded {len(DOCUMENTS_DB)} documents from {DOCUMENTS_DIR}.")

# Dependency to ensure documents are loaded
# FastAPI's Depends will call this before requests to endpoints that use it.
# This is better than a global DOCUMENTS_DB check in each route.
# However, for MCP server that might run once and load, a startup event is better.
# The model-context-protocol SDK might have its own way to handle startup events.
# For now, using a simple load on module import for simplicity,
# or a Depends if we want it per-request (less efficient for static data).

# Let's load documents when this module is first imported.
# This is a common pattern for data that should be available when the app starts.
# In FastAPI, you'd use a startup event handler on the main app object.
# Since this is a router, we can call it here.
load_documents_from_disk()

search_router = APIRouter()

def get_db():
    """Dependency that returns the loaded documents DB."""
    if not DOCUMENTS_DB and os.path.exists(DOCUMENTS_DIR):
        # Attempt to reload if DB is empty but dir exists (e.g., if files were added after initial load)
        # This is a simple reload strategy; more robust might be needed.
        logger.info("DOCUMENTS_DB is empty, attempting to reload documents.")
        load_documents_from_disk()
    return DOCUMENTS_DB


@search_router.get("/documents", response_model=ListDocumentsResponse)
async def list_all_documents(db: Dict[str, Dict[str, Any]] = Depends(get_db)):
    """Returns metadata of available documents."""
    if not db:
        # This case should ideally be handled by get_db or initial load,
        # but as a fallback:
        # raise HTTPException(status_code=404, detail="No documents loaded. Check server logs and data/documents directory.")
        # The ListDocumentsResponse schema allows for empty list and success=True
        logger.warning("Listing documents, but document DB is empty.")
        return ListDocumentsResponse(success=True, documents=[])

    metadata_list = [
        DocumentMetadata(
            id=doc_data["id"],
            filename=doc_data["filename"],
            content_preview=doc_data["content_preview"]
        ) for doc_id, doc_data in db.items()
    ]
    return ListDocumentsResponse(success=True, documents=metadata_list)

@search_router.get("/documents/{doc_id}", response_model=DocumentContentResponse)
async def get_specific_document(doc_id: str, db: Dict[str, Dict[str, Any]] = Depends(get_db)):
    """Returns content of a specific document."""
    doc_data = db.get(doc_id)
    if doc_data:
        return DocumentContentResponse(
            success=True,
            id=doc_data["id"],
            filename=doc_data["filename"],
            content=doc_data["full_content"]
        )
    else:
        # Return a structured error within the response model
        # Or raise HTTPException:
        # raise HTTPException(status_code=404, detail="Document not found")
        logger.warning(f"Document with ID '{doc_id}' not found.")
        return DocumentContentResponse(success=False, error="Document not found")


@search_router.post("/search", response_model=SearchResponse)
async def search_all_documents(request: SearchRequest, db: Dict[str, Dict[str, Any]] = Depends(get_db)):
    """
    Performs a simple keyword search on the documents.
    """
    if not db:
        logger.warning("Search request, but document DB is empty.")
        return SearchResponse(success=True, results=[]) # Empty results if no docs

    search_query = request.query.lower()
    if not search_query.strip():
        # raise HTTPException(status_code=400, detail="'query' must be a non-empty string")
        return SearchResponse(success=False, error="'query' must be a non-empty string")


    results = []
    search_terms = search_query.split()

    for doc_id, doc_data in db.items():
        score = 0
        content_lower = doc_data["full_content"].lower()
        for term in search_terms:
            score += content_lower.count(term) # Simple term frequency

        if score > 0:
            results.append(SearchResultItem(
                id=doc_data["id"],
                filename=doc_data["filename"],
                score=float(score), # Ensure score is float
                content_preview=doc_data["content_preview"]
            ))

    results.sort(key=lambda x: x.score, reverse=True)
    logger.info(f"Search for '{request.query}' yielded {len(results)} results.")
    return SearchResponse(success=True, results=results)

# Example of how this router would be included in an MCPApplication:
"""
from mcpserver import MCPApplication
from mcp_servers.search_mcp import search_router, load_documents_from_disk

# Create the main MCP application instance
app = MCPApplication(
    name="SearchMCP",
    description="MCP for searching text documents.",
    version="1.0.0",
    api_prefix="/search_service"
)

# Include the search router
app.include_router(search_router, prefix="/v1", tags=["Document Search"])

# It's good practice to load data on startup if using FastAPI/MCPApplication
@app.on_event("startup")
async def startup_event():
    print("Search MCP starting up. Loading documents...")
    load_documents_from_disk() # Ensure documents are loaded
    print(f"Documents loaded: {len(DOCUMENTS_DB)} items.")

# To run with uvicorn:
# uvicorn main_search_server:app --host 0.0.0.0 --port 5003
"""

logger.info("Search MCP router defined.")
