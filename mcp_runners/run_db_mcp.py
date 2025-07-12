import uvicorn
from mcpserver import MCPApplication # type: ignore
from mcp_servers.db_mcp import db_router, DATABASE_PATH, logger as db_logger # Import the router

# Create the main MCP application instance for the DB MCP
db_app = MCPApplication(
    name="DatabaseMCP",
    description="MCP for querying a CSV database.",
    version="1.0.0",
    # api_prefix="/db_service" # Optional: if you want all routes under /db_service
)

# Include the DB router
# If api_prefix is set on MCPApplication, this prefix is appended.
# e.g. /db_service/v1/query
# If no api_prefix on app, then /v1/query
# For current agent setup (expects /query), let's use root path for the router for now
# or ensure agent URLs are updated.
# To match agent expectation of http://localhost:5001/query:
db_app.include_router(db_router, prefix="", tags=["Database Queries"])
# Or, if you want /v1/query, then agent URL must be http://localhost:5001/v1/query

import os # Import os for path checking

@db_app.on_event("startup")
async def startup_event():
    db_logger.info(f"Database MCP starting up. Checking DB at {DATABASE_PATH}...")
    # Any other startup logic for DB MCP
    if not os.path.exists(DATABASE_PATH):
        db_logger.error(f"CRITICAL: Database file {DATABASE_PATH} not found on startup!")
    else:
        db_logger.info(f"Database file {DATABASE_PATH} confirmed on startup.")

if __name__ == "__main__":
    # This allows running this specific MCP server directly
    # Ensure PYTHONPATH is set so 'mcp_servers.db_mcp' and 'schemas' can be found
    # Example: export PYTHONPATH=$PYTHONPATH:$(pwd) from project root
    # Or run as module: python -m mcp_runners.run_db_mcp
    # import os # os is now imported at the top
    print(f"Attempting to run DB MCP on port 5001. Ensure schemas.py is in PYTHONPATH.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

    # Check if schemas.py is accessible (simple check)
    try:
        import schemas
        print("schemas.py found successfully.")
    except ImportError:
        print("ERROR: schemas.py not found. Please ensure the project root is in PYTHONPATH.")
        print("Example: export PYTHONPATH=${PYTHONPATH}:`pwd` (from project root)")
        exit(1)

    uvicorn.run(db_app, host="0.0.0.0", port=5001, log_level="info")
