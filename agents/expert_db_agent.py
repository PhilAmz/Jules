import httpx
import json # For more complex query generation if needed
from typing import Dict, List, Optional, Any
from schemas import DBQueryRequest, DBQueryResponse # Assuming schemas.py is in PYTHONPATH or same dir level

# Configuration for the DB MCP
DB_MCP_URL = "http://localhost:5001" # Assuming db_mcp.py runs on port 5001

class ExpertDBAgent:
    def __init__(self, mcp_url: str = DB_MCP_URL):
        self.mcp_url = mcp_url
        self.client = httpx.AsyncClient(timeout=30.0) # Using async client for potential future async orchestrator

    async def _make_request(self, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper function to make requests to the DB MCP."""
        try:
            response = await self.client.post(f"{self.mcp_url}{endpoint}", json=payload)
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses
            return response.json()
        except httpx.HTTPStatusError as e:
            # Log error or handle specific statuses
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            # Try to parse error from response if JSON
            try:
                error_details = e.response.json()
                return {"success": False, "error": error_details.get("error", e.response.text), "data": None}
            except json.JSONDecodeError:
                return {"success": False, "error": e.response.text, "data": None}
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            return {"success": False, "error": f"Request failed: {str(e)}", "data": None}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"success": False, "error": "Failed to decode JSON response from MCP.", "data": None}


    async def query_database(self, natural_language_query: str) -> DBQueryResponse:
        """
        Processes a natural language query, translates it to a structured DB query,
        and fetches results from the DB MCP.

        For this example, the translation will be very basic. A real implementation
        would involve more sophisticated NLP or a structured input for filters/columns.
        """
        # Basic "translation" - This is where more complex NLP would go.
        # Example: "find all data for SensorA and show ID and Value"
        # This is highly simplified and would need a robust parser for general use.
        filters = None
        select_columns = None

        # Extremely simple keyword-based parsing for demonstration
        if "for " in natural_language_query.lower():
            try:
                # e.g. "for SensorA" -> filters = {"Name": "SensorA"}
                # This assumes a very specific phrasing.
                entity_part = natural_language_query.lower().split("for ", 1)[1].split(" ",1)[0].strip()
                # This is a massive simplification. In a real system, you'd identify column and value.
                # Let's assume for now "for SensorA" means filter by Name = SensorA
                if entity_part: # Basic check
                    filters = {"Name": entity_part.capitalize()} # Assuming 'Name' is the column and needs capitalization
            except IndexError:
                pass # Could not parse entity

        if "show " in natural_language_query.lower() or "select " in natural_language_query.lower():
            try:
                # e.g. "show ID and Value" or "select ID, Value"
                columns_part = ""
                if "show " in natural_language_query.lower():
                    columns_part = natural_language_query.lower().split("show ", 1)[1]
                elif "select " in natural_language_query.lower():
                    columns_part = natural_language_query.lower().split("select ", 1)[1]

                # Remove any trailing instructions like "from ..."
                columns_part = columns_part.split(" from ",1)[0]

                # Split by "and" or ","
                raw_cols = []
                if " and " in columns_part:
                    raw_cols = [col.strip().upper() for col in columns_part.split(" and ")]
                elif "," in columns_part:
                    raw_cols = [col.strip().upper() for col in columns_part.split(",")]
                else: # Single column
                    raw_cols = [columns_part.strip().upper()]

                # This is a placeholder. Real column names from CSV are case-sensitive.
                # For now, we'll assume these map directly. A better agent would know schema.
                # For the sample_db.csv, columns are ID, Name, Value, Timestamp
                valid_cols = {"ID", "NAME", "VALUE", "TIMESTAMP"} # Use uppercase for matching
                select_columns = [col for col in raw_cols if col in valid_cols]
                if not select_columns and raw_cols: # if user specified cols but none were valid
                    print(f"Warning: Specified columns '{raw_cols}' not recognized or empty after validation. Returning all columns.")
                    select_columns = None # Default to all if parsing fails to find valid ones

            except IndexError:
                pass # Could not parse columns

        print(f"Translated query: Filters={filters}, Select Columns={select_columns}")

        query_payload = DBQueryRequest(filters=filters, select_columns=select_columns)

        response_json = await self._make_request("/query", payload=query_payload.model_dump(exclude_none=True))

        # _make_request ensures response_json is a dict that can be unpacked into DBQueryResponse
        # for both successful responses from the FastAPI server and fabricated error structures
        # from _make_request itself in case of HTTP/network issues.
        try:
            # The FastAPI MCP endpoint for /query directly returns a JSON that matches DBQueryResponse.
            # If _make_request caught an HTTP error, it also structures its return to match.
            return DBQueryResponse(**response_json)
        except Exception as e: # Fallback if response_json doesn't match the model (e.g., missing fields)
            print(f"Error parsing DBQueryResponse from MCP response: {response_json}, error: {e}")
            # Ensure this path still populates all required fields of DBQueryResponse if some are missing
            # Default Pydantic behavior might handle missing Optionals correctly.
            # If 'success' is missing, this might fail. A robust way:
            processed_error_response = {
                "success": False,
                "data": None,
                "error": f"Invalid response structure from DB MCP: {e}. Original: {str(response_json)[:200]}"
            }
            # Merge with whatever was in response_json, prioritizing our error structure.
            if isinstance(response_json, dict):
                processed_error_response.update(response_json) # Update with any existing fields
                processed_error_response["success"] = False # Ensure success is false
                if "error" not in response_json: # Ensure error field is populated
                     processed_error_response["error"] = f"Invalid response structure from DB MCP: {e}. Original: {str(response_json)[:200]}"

            return DBQueryResponse(**processed_error_response)


    async def close(self):
        """Closes the HTTP client."""
        await self.client.aclose()

# Example Usage (for testing the agent directly)
async def main():
    agent = ExpertDBAgent()

    # Test cases
    # query1_nl = "get all data"
    # print(f"\nTesting query: {query1_nl}")
    # response1 = await agent.query_database(query1_nl)
    # print(f"Response 1: {response1.model_dump_json(indent=2)}")

    # query2_nl = "find all data for SensorA and show ID and Value"
    # print(f"\nTesting query: {query2_nl}")
    # response2 = await agent.query_database(query2_nl)
    # print(f"Response 2: {response2.model_dump_json(indent=2)}")

    # query3_nl = "show Name, Timestamp for SensorC" # Note: column order might differ based on dict
    # print(f"\nTesting query: {query3_nl}")
    # response3 = await agent.query_database(query3_nl)
    # print(f"Response 3: {response3.model_dump_json(indent=2)}")

    # query4_nl = "get everything for SensorB"
    # print(f"\nTesting query: {query4_nl}")
    # response4 = await agent.query_database(query4_nl)
    # print(f"Response 4: {response4.model_dump_json(indent=2)}")

    # query5_nl = "show me ID for NonExistentSensor"
    # print(f"\nTesting query: {query5_nl}")
    # response5 = await agent.query_database(query5_nl)
    # print(f"Response 5: {response5.model_dump_json(indent=2)}")

    # query6_nl = "show blarg and wumph" # Invalid columns
    # print(f"\nTesting query: {query6_nl}")
    # response6 = await agent.query_database(query6_nl)
    # print(f"Response 6: {response6.model_dump_json(indent=2)}")


    await agent.close()

if __name__ == "__main__":
    # This part is for direct testing of the agent.
    # You would need to run the db_mcp.py server separately.
    # import asyncio
    # asyncio.run(main())
    pass
