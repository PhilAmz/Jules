import asyncio
import os
from agents.orchestrator_agent import OrchestratorAgent
from schemas import OrchestratorRequest

async def run_orchestration():
    print("Starting Orchestrator Agent example...")
    print("IMPORTANT: Ensure all MCP servers (DB, Code Execution, Search) are running separately.")
    print("You can run them using:")
    print("  python -m mcp_runners.run_db_mcp")
    print("  python -m mcp_runners.run_code_execution_mcp")
    print("  python -m mcp_runners.run_search_mcp")
    print("Ensure you have run 'export PYTHONPATH=${PYTHONPATH}:`pwd`' from the project root first.\n")

    orchestrator = OrchestratorAgent()

    # --- Define User Prompt ---
    # You can change this prompt to test different scenarios.
    # user_prompt = "query database for SensorA and show ID and Value"
    # user_prompt = "list documents and search for bananas"
    # user_prompt = "execute code: print('Hello from main.py test!')\nresult = 123 * 2"
    # user_prompt = "find data for SensorC then search for oranges and also run this python: result = 10*5"
    user_prompt = "list documents, then get document doc1, then search for apples in all documents, then find data for SensorB, and finally run this python code: result = 'done with all tasks'"


    print(f"--- Processing User Prompt ---")
    print(f"Prompt: \"{user_prompt}\"")
    print("-" * 30)

    request = OrchestratorRequest(user_prompt=user_prompt)

    try:
        response = await orchestrator.process_prompt(request)

        print("\n--- Orchestrator's Plan ---")
        if response.plan:
            print(response.plan.model_dump_json(indent=2, exclude_none=True))
        else:
            print("No plan was generated.")

        print("\n--- Orchestrator's Final Answer ---")
        print(response.final_answer)

    except Exception as e:
        print(f"An error occurred during orchestration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Closing expert agent connections ---")
        await orchestrator.close_expert_agents()
        print("Orchestration finished.")

if __name__ == "__main__":
    # Check if schemas.py is accessible (simple check)
    # This is important because agents and schemas are typically at the root or structured as a package.
    try:
        import schemas
        print("main.py: schemas.py found successfully.")
    except ImportError:
        print("main.py: ERROR: schemas.py not found. Please ensure the project root is in PYTHONPATH.")
        print("Example: export PYTHONPATH=${PYTHONPATH}:`pwd` (from project root)")
        print("Alternatively, structure your project as a Python package.")
        # exit(1) # Don't exit if just running, let it fail in agent import

    asyncio.run(run_orchestration())
