# Agent-Based Orchestration System with MCPs

This project demonstrates an agent-based system where an Orchestrator Agent delegates tasks to specialized Expert Agents. These Expert Agents interact with Managed Component Provider (MCP) servers that offer specific functionalities like database querying, code execution, and document search. The communication is structured using Pydantic models, and the MCP servers are built using the `model-context-protocol` SDK (which leverages FastAPI).

## Project Structure

-   `agents/`: Contains the implementations for the Orchestrator Agent and Expert Agents.
    -   `orchestrator_agent.py`: The main orchestrator.
    -   `expert_db_agent.py`: Agent for interacting with the Database MCP.
    -   `expert_code_agent.py`: Agent for interacting with the Code Execution MCP.
    -   `expert_search_agent.py`: Agent for interacting with the Search MCP.
-   `mcp_servers/`: Contains the router definitions for each MCP service.
    -   `db_mcp.py`: Router for CSV database functionalities.
    -   `code_execution_mcp.py`: Router for Python code execution.
    -   `search_mcp.py`: Router for document search functionalities.
-   `mcp_runners/`: Contains scripts to run each MCP server.
    -   `run_db_mcp.py`: Runs the Database MCP.
    -   `run_code_execution_mcp.py`: Runs the Code Execution MCP.
    -   `run_search_mcp.py`: Runs the Search MCP.
-   `data/`: Contains sample data used by the MCPs.
    -   `database/sample_db.csv`: Sample CSV database.
    -   `documents/`: Contains sample text files for the search MCP.
-   `schemas.py`: Defines Pydantic models for structured data exchange between agents and MCPs, and for internal orchestration logic.
-   `main.py`: Example script to run the Orchestrator Agent with a sample prompt.
-   `requirements.txt`: Python dependencies.
-   `AGENTS.md`: This file.

## Setup Instructions

1.  **Clone the Repository (if applicable)**
    ```bash
    # git clone ...
    # cd your_project_directory
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Set PYTHONPATH**
    For the application to correctly find modules (like `schemas.py`, `agents.`, `mcp_servers.`), add the project's root directory to your `PYTHONPATH`. From the root directory of this project:
    ```bash
    export PYTHONPATH=${PYTHONPATH}:`pwd`
    ```
    *(On Windows, you might need to set this environment variable through system settings or use `set PYTHONPATH=%PYTHONPATH%;%CD%` in Command Prompt, or `$env:PYTHONPATH += ";${pwd}"` in PowerShell)*

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Running the System

The system requires three MCP servers to be running independently, and then the `main.py` script can be executed to interact with the Orchestrator Agent.

1.  **Start the Database MCP Server**
    Open a new terminal, ensure your virtual environment is activated and `PYTHONPATH` is set, then run:
    ```bash
    python -m mcp_runners.run_db_mcp
    ```
    This server will run on `http://localhost:5001`.

2.  **Start the Code Execution MCP Server**
    Open another new terminal, ensure your virtual environment is activated and `PYTHONPATH` is set, then run:
    ```bash
    python -m mcp_runners.run_code_execution_mcp
    ```
    This server will run on `http://localhost:5002`.

3.  **Start the Search MCP Server**
    Open a third new terminal, ensure your virtual environment is activated and `PYTHONPATH` is set, then run:
    ```bash
    python -m mcp_runners.run_search_mcp
    ```
    This server will run on `http://localhost:5003`.

4.  **Run the Main Orchestrator Application**
    Once all three MCP servers are running, open a fourth terminal, ensure your virtual environment is activated and `PYTHONPATH` is set, then run:
    ```bash
    python main.py
    ```
    This script will send a sample prompt to the Orchestrator Agent, which will then plan and execute tasks using the running MCPs via the Expert Agents. The output, including the plan and final answer, will be printed to the console.

    You can modify the `user_prompt` variable in `main.py` to test different scenarios.

## How It Works

1.  `main.py` sends a user prompt to the `OrchestratorAgent`.
2.  The `OrchestratorAgent` (specifically, its `_generate_plan` method) parses the prompt into a series of tasks. This simple planner uses keyword matching.
3.  The orchestrator determines which Expert Agent is needed for each task.
4.  Tasks are executed (currently in parallel for all generated tasks). Expert Agents make HTTP requests to their respective MCP servers.
    -   `ExpertDBAgent` calls the Database MCP (`/query`).
    -   `ExpertCodeAgent` calls the Code Execution MCP (`/execute`).
    -   `ExpertSearchAgent` calls the Search MCP (`/documents`, `/documents/{id}`, `/search`).
5.  MCP servers process the requests and return JSON responses.
6.  Expert Agents parse these responses using Pydantic models.
7.  The `OrchestratorAgent` collects the results and formulates a final summary.

## Notes for Agent Developers (You!)

-   **PYTHONPATH**: Remember the `PYTHONPATH` setup. It's crucial for imports to work correctly when running modules like `mcp_runners` or `main.py`.
-   **MCP Server URLs**: Agent classes have hardcoded default URLs for the MCPs (e.g., `DB_MCP_URL = "http://localhost:5001"`). If you change the ports or deploy these MCPs elsewhere, these URLs in the agent files (`agents/*_agent.py` and `agents/orchestrator_agent.py`) will need to be updated, or a configuration management system should be implemented.
-   **Error Handling**: The system includes basic error handling. Check server logs and `main.py` output for errors.
-   **Schema Consistency**: Communication relies on Pydantic models defined in `schemas.py`. Ensure any changes to request/response structures are reflected here and in the corresponding agent and MCP code.
-   **Simplified Planner**: The planner in `OrchestratorAgent` is very basic. For more complex scenarios, this would be the primary area for enhancement, likely involving an LLM for more robust understanding, planning, and dependency management.
-   **Sequential vs. Parallel Execution**: The current orchestrator runs all identified tasks in one parallel batch. A more advanced version would analyze dependencies in the `Plan.execution_order` to run sequential groups of tasks.
-   **MCP Server Router Prefixes**: The MCP runner scripts currently include their FastAPI routers with an empty prefix (e.g., `db_app.include_router(db_router, prefix="")`). This means endpoints like `/query` are directly available at `http://localhost:5001/query`. If you add prefixes in the runner scripts (e.g., `prefix="/v1"`), ensure the agent URLs in `agents/*_agent.py` are updated accordingly (e.g., to `http://localhost:5001/v1/query`).

This setup provides a foundational, albeit simplified, example of an orchestrated multi-agent system using the `model-context-protocol` paradigm for its service components.
