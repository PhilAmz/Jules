import os
import csv
import logging
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
# Assuming schemas.py is one level up from mcp_servers, and accessible.
# If agents and mcp_servers are siblings, and schemas.py is at root:
# from ..schemas import DBQueryRequest, DBQueryResponse
# For now, let's assume schemas.py is findable in PYTHONPATH or we redefine necessary models here.
# To keep it self-contained for now, let's redefine just for this MCP or ensure path.
# For simplicity, let's assume PYTHONPATH is set up or schemas.py is moved/copied.
# This path might need adjustment based on final project structure and how it's run.
try:
    from schemas import DBQueryRequest, DBQueryResponse, DBRow
except ImportError:
    # Fallback if schemas.py is not directly accessible via PYTHONPATH
    # This is not ideal but can help during isolated development/testing of MCP
    print("Warning: schemas.py not found in default PYTHONPATH. Redefining local schemas for DB MCP.")
    class DBQueryRequest(BaseModel):
        filters: Optional[Dict[str, Any]] = None
        select_columns: Optional[List[str]] = None

    class DBRow(BaseModel): # Placeholder, actual fields depend on query
        pass

    class DBQueryResponse(BaseModel):
        success: bool = True
        data: Optional[List[Dict[str, Any]]] = None
        error: Optional[str] = None


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Database path - relative to this file's location, then up and into data/
# __file__ is mcp_servers/db_mcp.py
# ../ is project root
# ../data/database/sample_db.csv
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'database', 'sample_db.csv')
logger.info(f"Database path configured to: {DATABASE_PATH}")

# Create a router for the DB service
db_router = APIRouter()

def query_csv_data(file_path: str, filters: Optional[Dict[str, Any]] = None, select_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Core logic to query a CSV file.
    - file_path: Path to the CSV file.
    - filters: A dictionary where keys are column names and values are the values to filter by.
    - select_columns: A list of column names to return. If None, returns all columns.
    Returns a list of dictionaries representing the rows.
    Raises FileNotFoundError if csv file not found.
    """
    data = []
    if not os.path.exists(file_path):
        logger.error(f"Database file not found at {file_path}")
        raise FileNotFoundError(f"Database file not found at {file_path}")

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            match = True
            if filters:
                for key, value in filters.items():
                    # Ensure case-insensitive matching for filter keys if needed, though CSV headers are exact.
                    # For values, ensure type consistency or convert if necessary.
                    if row.get(key) != str(value): # Basic string comparison for demo
                        match = False
                        break
            if match:
                if select_columns:
                    # Filter for existing columns to prevent KeyErrors if select_columns has invalid names
                    selected_row = {col: row.get(col) for col in select_columns if col in row}
                    if selected_row: # Only add if there are any valid selected columns after filtering
                        data.append(selected_row)
                else:
                    data.append(row)
    return data

@db_router.post("/query", response_model=DBQueryResponse)
async def handle_db_query(request: DBQueryRequest) -> DBQueryResponse:
    """
    Handles querying the CSV database.
    """
    logger.info(f"Received query request: filters={request.filters}, select_columns={request.select_columns}")
    try:
        # Basic validation for filters and select_columns (Pydantic already does some)
        if request.filters is not None and not isinstance(request.filters, dict):
            raise HTTPException(status_code=400, detail="filters must be a dictionary or null")
        if request.select_columns is not None and not isinstance(request.select_columns, list):
            raise HTTPException(status_code=400, detail="select_columns must be a list or null")
        if request.select_columns is not None and not all(isinstance(col, str) for col in request.select_columns):
             raise HTTPException(status_code=400, detail="all elements in select_columns must be strings")

        query_result = query_csv_data(
            DATABASE_PATH,
            filters=request.filters,
            select_columns=request.select_columns
        )
        logger.info(f"Query successful, {len(query_result)} rows returned.")
        return DBQueryResponse(success=True, data=query_result)
    except FileNotFoundError as e:
        logger.error(f"Database file not found: {e}")
        # Return a 500 for server-side file issues, or 404 if it's more like a missing resource
        # For a DB, if the configured DB file is missing, it's a server error.
        # The MCP SDK might have its own way of propagating these.
        # Using HTTPException for now as it's standard FastAPI.
        # The model-context-protocol server should handle these and package them.
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred during DB query.")
        # Propagate other errors too
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run this server with model-context-protocol's runner,
# you would typically have a main script that initializes the MCPApplication
# and includes this router.
# For example, in a main_db_server.py:
"""
from mcpserver import MCPApplication
from mcp_servers.db_mcp import db_router # Assuming this file is db_mcp.py

app = MCPApplication(
    name="DatabaseMCP",
    description="MCP for querying a CSV database.",
    version="1.0.0",
    api_prefix="/db_service" # Example prefix
)

app.include_router(db_router, prefix="/v1", tags=["Database Queries"])

# To run with uvicorn:
# uvicorn main_db_server:app --host 0.0.0.0 --port 5001
"""

# This check ensures the DB file exists when the module is loaded (optional)
if not os.path.exists(DATABASE_PATH):
    logger.warning(f"Initial check: Database file {DATABASE_PATH} not found. Server will fail queries until it's available.")
else:
    logger.info(f"Initial check: Database file found at {DATABASE_PATH}.")

# Note: The actual server startup (MCPApplication) will be handled by a main runner script
# or by the model-context-protocol CLI if it provides one. This file now primarily defines the router.
# The previous Flask `app.run()` is removed.
