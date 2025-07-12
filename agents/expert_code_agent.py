import httpx
import json
from typing import Dict, Optional, Any
from schemas import CodeExecutionRequest, CodeExecutionResponse # Assuming schemas.py is in PYTHONPATH

# Configuration for the Code Execution MCP
CODE_EXEC_MCP_URL = "http://localhost:5002" # Assuming code_execution_mcp.py runs on port 5002

class ExpertCodeAgent:
    def __init__(self, mcp_url: str = CODE_EXEC_MCP_URL):
        self.mcp_url = mcp_url
        self.client = httpx.AsyncClient(timeout=60.0) # Longer timeout for potentially long-running code

    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Helper function to make requests to the Code Execution MCP."""
        try:
            response = await self.client.post(f"{self.mcp_url}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            try:
                error_details = e.response.json()
                return {"success": False, "error": error_details.get("error", e.response.text), "stdout": "", "result": None}
            except json.JSONDecodeError:
                 return {"success": False, "error": e.response.text, "stdout": "", "result": None}
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            return {"success": False, "error": f"Request failed: {str(e)}", "stdout": "", "result": None}
        except json.JSONDecodeError as e: # Should not happen if MCP always returns valid JSON
            print(f"JSON decode error: {e}")
            return {"success": False, "error": "Failed to decode JSON response from Code Execution MCP.", "stdout": "", "result": None}


    async def execute_code_snippet(self, code_snippet: str) -> CodeExecutionResponse:
        """
        Sends a Python code snippet to the Code Execution MCP for execution.
        """
        if not isinstance(code_snippet, str):
            return CodeExecutionResponse(success=False, error="Code snippet must be a string.")

        request_payload = CodeExecutionRequest(code=code_snippet)

        response_json = await self._make_request("/execute", payload=request_payload.model_dump())

        # The MCP should always return a response that fits CodeExecutionResponse structure,
        # even for errors during code execution (success=False, error="...").
        # The _make_request method also ensures a compatible dict structure on client/HTTP errors.
        try:
            return CodeExecutionResponse(**response_json)
        except Exception as e: # Fallback for unexpected response structure after all
            print(f"Error parsing CodeExecutionResponse from MCP response: {response_json}, error: {e}")
            return CodeExecutionResponse(success=False, error=f"Invalid response structure from MCP: {e}", stdout=None, result=None)

    async def close(self):
        """Closes the HTTP client."""
        await self.client.aclose()

# Example Usage (for testing the agent directly)
async def main():
    agent = ExpertCodeAgent()

    # Test case 1: Simple print
    code1 = "print('Hello from executed code!')\nresult = 42"
    print(f"\nExecuting code: \n{code1}")
    response1 = await agent.execute_code_snippet(code1)
    print(f"Response 1: {response1.model_dump_json(indent=2)}")

    # Test case 2: Code with an error
    code2 = "print('This will print')\nx = 1 / 0\nprint('This will not print')"
    print(f"\nExecuting code with error: \n{code2}")
    response2 = await agent.execute_code_snippet(code2)
    print(f"Response 2: {response2.model_dump_json(indent=2)}")

    # Test case 3: Code that defines 'result'
    code3 = "a = 10\nb = 20\nresult = a * b"
    print(f"\nExecuting code with result: \n{code3}")
    response3 = await agent.execute_code_snippet(code3)
    print(f"Response 3: {response3.model_dump_json(indent=2)}")

    # Test case 4: Invalid input type (not a string)
    # print(f"\nExecuting invalid input type:")
    # response4 = await agent.execute_code_snippet(12345) # type: ignore
    # print(f"Response 4: {response4.model_dump_json(indent=2)}")


    await agent.close()

if __name__ == "__main__":
    # This part is for direct testing of the agent.
    # You would need to run the code_execution_mcp.py server separately.
    # import asyncio
    # asyncio.run(main())
    pass
