import uvicorn
from mcpserver import MCPApplication # type: ignore
from mcp_servers.search_mcp import search_router, load_documents_from_disk, DOCUMENTS_DB, DOCUMENTS_DIR, logger as search_logger
import os # For path check

# Create the main MCP application instance for the Search MCP
search_app = MCPApplication(
    name="SearchMCP",
    description="MCP for searching text documents.",
    version="1.0.0",
    # api_prefix="/search_service"
)

# To match agent expectation of http://localhost:5003/documents etc:
search_app.include_router(search_router, prefix="", tags=["Document Search"])

@search_app.on_event("startup")
async def startup_event():
    search_logger.info(f"Search MCP starting up. Document dir: {DOCUMENTS_DIR}")
    # Documents are loaded when search_mcp module is imported.
    # This event can confirm or trigger a reload if necessary.
    if not os.path.exists(DOCUMENTS_DIR):
        search_logger.error(f"CRITICAL: Documents directory {DOCUMENTS_DIR} not found on startup!")
    elif not DOCUMENTS_DB:
        search_logger.warning(f"Documents DB is empty on startup, attempting reload from {DOCUMENTS_DIR}.")
        load_documents_from_disk() # Explicitly call load here too
    search_logger.info(f"Search MCP startup: {len(DOCUMENTS_DB)} documents loaded.")


if __name__ == "__main__":
    print(f"Attempting to run Search MCP on port 5003. Ensure schemas.py is in PYTHONPATH.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

    try:
        import schemas
        print("schemas.py found successfully.")
    except ImportError:
        print("ERROR: schemas.py not found. Please ensure the project root is in PYTHONPATH.")
        exit(1)
    uvicorn.run(search_app, host="0.0.0.0", port=5003, log_level="info")
