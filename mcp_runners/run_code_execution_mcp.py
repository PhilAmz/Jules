import uvicorn
from mcpserver import MCPApplication # type: ignore
from mcp_servers.code_execution_mcp import code_exec_router, logger as code_exec_logger

# Create the main MCP application instance for the Code Execution MCP
code_exec_app = MCPApplication(
    name="CodeExecutionMCP",
    description="MCP for executing Python code snippets.",
    version="1.0.0",
    # api_prefix="/code_exec_service"
)

# To match agent expectation of http://localhost:5002/execute:
code_exec_app.include_router(code_exec_router, prefix="", tags=["Code Execution"])

@code_exec_app.on_event("startup")
async def startup_event():
    code_exec_logger.info("Code Execution MCP starting up...")
    # Any other startup logic

if __name__ == "__main__":
    import os
    print(f"Attempting to run Code Execution MCP on port 5002. Ensure schemas.py is in PYTHONPATH.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    try:
        import schemas
        print("schemas.py found successfully.")
    except ImportError:
        print("ERROR: schemas.py not found. Please ensure the project root is in PYTHONPATH.")
        exit(1)
    uvicorn.run(code_exec_app, host="0.0.0.0", port=5002, log_level="info")
