import asyncio
import uuid
from typing import Dict, List, Any, Coroutine

from schemas import (
    OrchestratorRequest, OrchestratorResponse,
    Plan, Task,
    DBQueryResponse, CodeExecutionResponse, SearchResponse,
    ListDocumentsResponse, DocumentContentResponse
)
from agents.expert_db_agent import ExpertDBAgent
from agents.expert_code_agent import ExpertCodeAgent
from agents.expert_search_agent import ExpertSearchAgent

# URLs for the MCPs - could be loaded from config
DB_MCP_URL = "http://localhost:5001"
CODE_EXEC_MCP_URL = "http://localhost:5002"
SEARCH_MCP_URL = "http://localhost:5003"

class OrchestratorAgent:
    def __init__(self):
        self.db_agent = ExpertDBAgent(mcp_url=DB_MCP_URL)
        self.code_agent = ExpertCodeAgent(mcp_url=CODE_EXEC_MCP_URL)
        self.search_agent = ExpertSearchAgent(mcp_url=SEARCH_MCP_URL)

    async def _generate_plan(self, user_prompt: str) -> Plan:
        """
        Generates a basic plan (todo list) from the user prompt.
        This is a highly simplified planner for demonstration.
        A real orchestrator would use an LLM or more sophisticated logic here.
        """
        tasks = []
        task_id_counter = 1

        # Simple keyword-based task generation
        prompt_lower = user_prompt.lower()

        if "database" in prompt_lower or "query" in prompt_lower or "find data" in prompt_lower:
            # Example: "query database for SensorA and show ID"
            # Extract the actual query part. This is still very naive.
            db_query_str = user_prompt # Pass the whole prompt for now, agent will parse
            if "query database " in prompt_lower:
                 db_query_str = user_prompt[user_prompt.lower().find("query database ") + len("query database "):]
            elif "find data " in prompt_lower:
                 db_query_str = user_prompt[user_prompt.lower().find("find data ") + len("find data "):]


            tasks.append(Task(
                id=str(uuid.uuid4()),
                description=f"Query database with: {db_query_str}",
                agent_type="db_query",
                params={"natural_language_query": db_query_str},
                dependencies=[]
            ))
            task_id_counter +=1

        if "search for" in prompt_lower or "look up" in prompt_lower:
            # Example: "search for apples"
            search_query_str = ""
            if "search for " in prompt_lower:
                search_query_str = user_prompt[prompt_lower.find("search for ") + len("search for "):]
            elif "look up " in prompt_lower:
                search_query_str = user_prompt[prompt_lower.find("look up ") + len("look up "):]

            if search_query_str:
                tasks.append(Task(
                    id=str(uuid.uuid4()),
                    description=f"Search documents for: {search_query_str}",
                    agent_type="search_documents",
                    params={"query": search_query_str},
                    dependencies=[] # Could depend on prior tasks if query involves their results
                ))
                task_id_counter +=1

        if "list documents" in prompt_lower:
            tasks.append(Task(
                id=str(uuid.uuid4()),
                description="List all available documents",
                agent_type="list_documents",
                params={},
                dependencies=[]
            ))
            task_id_counter += 1

        if "get document" in prompt_lower: # e.g. "get document doc1"
            doc_id_str = user_prompt[prompt_lower.find("get document ") + len("get document "):].split(" ")[0]
            if doc_id_str:
                tasks.append(Task(
                    id=str(uuid.uuid4()),
                    description=f"Get document with ID: {doc_id_str}",
                    agent_type="get_document",
                    params={"doc_id": doc_id_str},
                    dependencies=[]
                ))
                task_id_counter += 1

        if "execute code" in prompt_lower or "run this python" in prompt_lower:
            # Example: "execute code: print('hello')"
            # This is very naive, assumes code is clearly demarcated or simple.
            code_to_run = ""
            if "execute code: " in prompt_lower:
                 code_to_run = user_prompt[prompt_lower.find("execute code: ") + len("execute code: "):]
            elif "run this python: " in prompt_lower:
                 code_to_run = user_prompt[prompt_lower.find("run this python: ") + len("run this python: "):]

            if code_to_run:
                tasks.append(Task(
                    id=str(uuid.uuid4()),
                    description=f"Execute Python code: {code_to_run[:50]}...",
                    agent_type="code_exec",
                    params={"code_snippet": code_to_run},
                    dependencies=[]
                ))
                task_id_counter += 1

        # For this simple planner, all tasks run in parallel as a single group.
        # A real planner would analyze dependencies.
        execution_order = [[task.id for task in tasks]] if tasks else []

        return Plan(tasks=tasks, execution_order=execution_order)

    async def _execute_task(self, task: Task) -> Any:
        """Executes a single task using the appropriate expert agent."""
        task.status = "in_progress"
        result: Any = None
        try:
            if task.agent_type == "db_query":
                result = await self.db_agent.query_database(**task.params)
            elif task.agent_type == "code_exec":
                result = await self.code_agent.execute_code_snippet(**task.params)
            elif task.agent_type == "search_documents":
                result = await self.search_agent.search_documents(**task.params)
            elif task.agent_type == "list_documents":
                result = await self.search_agent.list_documents() # No params usually
            elif task.agent_type == "get_document":
                result = await self.search_agent.get_document_by_id(**task.params)
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")

            task.result = result.model_dump() if hasattr(result, 'model_dump') else result

            # Check success status from Pydantic models if they have it
            if hasattr(result, 'success') and not result.success:
                task.status = "failed"
                # The error should be within result.error
            else:
                task.status = "completed"

        except Exception as e:
            print(f"Error executing task {task.id} ({task.description}): {e}")
            task.status = "failed"
            task.result = {"error": str(e)}

        return task # Return the updated task object

    async def process_prompt(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Processes the user prompt: generates a plan, executes it, and returns results.
        """
        plan = await self._generate_plan(request.user_prompt)
        if not plan.tasks:
            return OrchestratorResponse(final_answer="No actionable tasks found in the prompt.", plan=plan)

        # Execute tasks based on the plan's execution order (simplified: one parallel group)
        # A real orchestrator would handle complex dependencies and sequential groups.

        all_task_coroutines: List[Coroutine[Any, Any, Task]] = []
        task_map: Dict[str, Task] = {t.id: t for t in plan.tasks}

        for task_group in plan.execution_order: # Groups of task IDs to run in parallel
            current_group_coroutines: List[Coroutine[Any, Any, Task]] = []
            for task_id in task_group:
                task_obj = task_map.get(task_id)
                if task_obj:
                    # Here, a more complex orchestrator would check dependencies from previous groups.
                    # For now, we assume dependencies are within the simplified planning.
                    current_group_coroutines.append(self._execute_task(task_obj))

            if current_group_coroutines:
                # Run tasks in the current group concurrently
                updated_tasks_in_group = await asyncio.gather(*current_group_coroutines)
                # Update the main task_map with results from this group
                for updated_task in updated_tasks_in_group:
                    task_map[updated_task.id] = updated_task

        # Update plan.tasks with the executed tasks that now have results and statuses
        plan.tasks = list(task_map.values())

        # For simplicity, the final answer is just a summary of task statuses and results.
        # A real agent might synthesize a more coherent natural language response.
        final_answer_parts = [f"Processed prompt: '{request.user_prompt}'"]
        for task in plan.tasks:
            final_answer_parts.append(
                f"Task '{task.description}': {task.status}. Result: {str(task.result)[:200]}..."
            )

        return OrchestratorResponse(
            plan=plan,
            final_answer="\n".join(final_answer_parts)
        )

    async def close_expert_agents(self):
        """Closes all expert agent clients."""
        await asyncio.gather(
            self.db_agent.close(),
            self.code_agent.close(),
            self.search_agent.close()
        )

# Example Usage (for testing the orchestrator directly)
async def main():
    orchestrator = OrchestratorAgent()

    # prompts_to_test = [
    #     "query database for SensorA and show ID and Value",
    #     "list documents and search for bananas", # Will generate two parallel tasks
    #     "execute code: print('Hello from orchestrator test!')",
    #     "find data for SensorC then search for oranges and also run this python: result = 10*5", # 3 tasks
    #     "get document doc1",
    #     "what can you do?" # Should result in no actionable tasks
    # ]

    # For a quick test, let's use one that might involve multiple steps
    test_prompt = "find data for SensorB, then search for document content mentioning 'SensorB', and also list all documents."
    # The current simple planner will make these parallel, not sequential.

    print(f"--- Testing prompt: '{test_prompt}' ---")
    request_obj = OrchestratorRequest(user_prompt=test_prompt)
    response = await orchestrator.process_prompt(request_obj)

    print("\n--- Orchestrator Plan ---")
    if response.plan:
        print(response.plan.model_dump_json(indent=2))

    print("\n--- Orchestrator Final Answer ---")
    print(response.final_answer)

    await orchestrator.close_expert_agents()

if __name__ == "__main__":
    # This part is for direct testing of the agent.
    # You would need to run ALL MCP servers (db_mcp, code_execution_mcp, search_mcp) separately.
    # import asyncio
    # asyncio.run(main())
    pass
