import io
import sys
import contextlib
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Assuming schemas.py is accessible.
try:
    from schemas import CodeExecutionRequest, CodeExecutionResponse
except ImportError:
    print("Warning: schemas.py not found in default PYTHONPATH. Redefining local schemas for Code Execution MCP.")
    class CodeExecutionRequest(BaseModel):
        code: str

    class CodeExecutionResponse(BaseModel):
        success: bool
        stdout: str | None = None
        result: Any | None = None
        error: str | None = None

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create a router for the Code Execution service
code_exec_router = APIRouter()

@contextlib.contextmanager
def capture_stdout_for_exec():
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout

@code_exec_router.post("/execute", response_model=CodeExecutionResponse)
async def handle_code_execution(request: CodeExecutionRequest) -> CodeExecutionResponse:
    """
    Executes a given Python code snippet.
    """
    logger.info(f"Received code execution request for code: \n{request.code[:200]}...") # Log snippet

    if not isinstance(request.code, str):
        # Pydantic should catch this, but an explicit check can be here too.
        # However, FastAPI relies on Pydantic for this, so this check is redundant if model is used.
        # For direct calls or other frameworks, it would be useful.
        # raise HTTPException(status_code=400, detail="'code' field must be a string")
        # This will be handled by Pydantic validation. If it gets here, it's a string.
        pass


    try:
        local_vars: Dict[str, Any] = {}

        with capture_stdout_for_exec() as captured_output:
            exec(request.code, {}, local_vars) # Execute in a controlled scope

        output_str = captured_output.getvalue()
        execution_result = local_vars.get("result", None) # Convention: code can set a 'result' variable

        logger.info(f"Code executed successfully. Stdout length: {len(output_str)}. Result: {type(execution_result)}")
        return CodeExecutionResponse(
            success=True,
            stdout=output_str,
            result=execution_result
        )
    except Exception as e:
        logger.error(f"Error during code execution: {e}", exc_info=True) # Log full traceback for server
        # For the client, just send the error message, not the full traceback for security.
        return CodeExecutionResponse(
            success=False,
            error=str(e),
            stdout="" # No stdout if exec failed partway
        )

# Example of how this router would be included in an MCPApplication:
"""
from mcpserver import MCPApplication
from mcp_servers.code_execution_mcp import code_exec_router

app = MCPApplication(
    name="CodeExecutionMCP",
    description="MCP for executing Python code snippets.",
    version="1.0.0",
    api_prefix="/code_exec_service"
)

app.include_router(code_exec_router, prefix="/v1", tags=["Code Execution"])

# To run with uvicorn:
# uvicorn main_code_exec_server:app --host 0.0.0.0 --port 5002
"""

logger.info("Code Execution MCP router defined.")
