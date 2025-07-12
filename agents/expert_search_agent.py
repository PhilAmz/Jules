import httpx
import json
from typing import Dict, List, Optional, Any
from schemas import ( # Assuming schemas.py is in PYTHONPATH
    ListDocumentsResponse, DocumentMetadata,
    DocumentContentResponse,
    SearchRequest, SearchResponse, SearchResultItem
)

# Configuration for the Search MCP
SEARCH_MCP_URL = "http://localhost:5003" # Assuming search_mcp.py runs on port 5003

class ExpertSearchAgent:
    def __init__(self, mcp_url: str = SEARCH_MCP_URL):
        self.mcp_url = mcp_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _make_get_request(self, endpoint: str) -> Dict[str, Any]:
        """Helper function to make GET requests to the Search MCP."""
        try:
            response = await self.client.get(f"{self.mcp_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP GET error occurred: {e.response.status_code} - {e.response.text}")
            try:
                error_details = e.response.json()
                # Adapt to expected Pydantic model structure for errors (e.g. ListDocumentsResponse)
                return {"success": False, "error": error_details.get("error", e.response.text)}
            except json.JSONDecodeError:
                return {"success": False, "error": e.response.text}
        except httpx.RequestError as e:
            print(f"Request GET error occurred: {e}")
            return {"success": False, "error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            print(f"JSON decode error for GET response: {e}")
            return {"success": False, "error": "Failed to decode JSON response from Search MCP."}

    async def _make_post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Helper function to make POST requests to the Search MCP."""
        try:
            response = await self.client.post(f"{self.mcp_url}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP POST error occurred: {e.response.status_code} - {e.response.text}")
            try:
                error_details = e.response.json()
                return {"success": False, "error": error_details.get("error", e.response.text)}
            except json.JSONDecodeError:
                return {"success": False, "error": e.response.text}
        except httpx.RequestError as e:
            print(f"Request POST error occurred: {e}")
            return {"success": False, "error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            print(f"JSON decode error for POST response: {e}")
            return {"success": False, "error": "Failed to decode JSON response from Search MCP."}

    async def list_documents(self) -> ListDocumentsResponse:
        """Lists all available documents from the Search MCP."""
        response_json = await self._make_get_request("/documents")
        try:
            # The Search MCP /documents endpoint returns a JSON matching ListDocumentsResponse
            return ListDocumentsResponse(**response_json)
        except Exception as e:
            print(f"Error parsing ListDocumentsResponse: {response_json}, error: {e}")
            return ListDocumentsResponse(success=False, documents=None, error=f"Invalid response: {e}")

    async def get_document_by_id(self, doc_id: str) -> DocumentContentResponse:
        """Retrieves a specific document by its ID from the Search MCP."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            return DocumentContentResponse(success=False, error="Document ID must be a non-empty string.")

        response_json = await self._make_get_request(f"/documents/{doc_id}")
        try:
            # The Search MCP /documents/{doc_id} endpoint returns JSON matching DocumentContentResponse
            return DocumentContentResponse(**response_json)
        except Exception as e:
            print(f"Error parsing DocumentContentResponse: {response_json}, error: {e}")
            return DocumentContentResponse(success=False, error=f"Invalid response for doc '{doc_id}': {e}")

    async def search_documents(self, query: str) -> SearchResponse:
        """Searches documents based on a query string using the Search MCP."""
        if not isinstance(query, str) or not query.strip():
            return SearchResponse(success=False, error="Search query must be a non-empty string.")

        request_payload = SearchRequest(query=query)
        response_json = await self._make_post_request("/search", payload=request_payload.model_dump())

        try:
            # The Search MCP /search endpoint returns JSON matching SearchResponse
            return SearchResponse(**response_json)
        except Exception as e:
            print(f"Error parsing SearchResponse: {response_json}, error: {e}")
            return SearchResponse(success=False, results=None, error=f"Invalid search response: {e}")

    async def close(self):
        """Closes the HTTP client."""
        await self.client.aclose()

# Example Usage (for testing the agent directly)
async def main():
    agent = ExpertSearchAgent()

    print("\n--- Listing documents ---")
    list_resp = await agent.list_documents()
    print(list_resp.model_dump_json(indent=2))

    # Assuming 'doc1' is a valid document ID from sample data
    if list_resp.success and list_resp.documents and any(doc.id == "doc1" for doc in list_resp.documents):
        print("\n--- Getting document 'doc1' ---")
        doc_resp = await agent.get_document_by_id("doc1")
        print(doc_resp.model_dump_json(indent=2))
    else:
        print("\nSkipping get_document_by_id for 'doc1' as it's not listed or list failed.")

    print("\n--- Getting non-existent document 'nonexistentdoc' ---")
    doc_resp_fail = await agent.get_document_by_id("nonexistentdoc")
    print(doc_resp_fail.model_dump_json(indent=2))


    print("\n--- Searching for 'apples' ---")
    search_resp_apples = await agent.search_documents("apples")
    print(search_resp_apples.model_dump_json(indent=2))

    print("\n--- Searching for 'information' ---")
    search_resp_info = await agent.search_documents("information")
    print(search_resp_info.model_dump_json(indent=2))

    print("\n--- Searching with empty query ---")
    search_resp_empty = await agent.search_documents("   ")
    print(search_resp_empty.model_dump_json(indent=2))

    await agent.close()

if __name__ == "__main__":
    # This part is for direct testing of the agent.
    # You would need to run the search_mcp.py server separately.
    # import asyncio
    # asyncio.run(main())
    pass
