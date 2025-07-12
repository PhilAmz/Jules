from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- DB MCP Schemas ---

class DBQueryFilters(BaseModel):
    """Flexible filters for DB Query"""
    pass  # Allows any field, effectively Dict[str, Any]

class DBQueryRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = Field(None, description="Dictionary of filters, e.g., {\"Name\": \"SensorA\"}")
    select_columns: Optional[List[str]] = Field(None, description="List of columns to return, e.g., [\"ID\", \"Value\"]")

class DBRow(BaseModel):
    """Represents a generic row from the database."""
    # This will be a flexible model, actual fields depend on the query and DB.
    # For CSV, keys are column headers.
    pass # Allows any field

class DBQueryResponse(BaseModel):
    success: bool = True
    data: Optional[List[Dict[str, Any]]] = None # Using Dict directly as rows can have varied columns
    error: Optional[str] = None

# --- Code Execution MCP Schemas ---

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="Python code snippet to execute.")

class CodeExecutionResponse(BaseModel):
    success: bool
    stdout: Optional[str] = None
    result: Optional[Any] = None # For any potential 'result' variable returned by the code
    error: Optional[str] = None

# --- Search MCP Schemas ---

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    content_preview: str

class ListDocumentsResponse(BaseModel):
    success: bool = True
    documents: Optional[List[DocumentMetadata]] = None
    error: Optional[str] = None

class DocumentContentRequest(BaseModel): # Though it's a GET, useful for agent's internal modeling
    doc_id: str

class DocumentContentResponse(BaseModel):
    success: bool = True
    id: Optional[str] = None
    filename: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string.")

class SearchResultItem(BaseModel):
    id: str
    filename: str
    score: float
    content_preview: str

class SearchResponse(BaseModel):
    success: bool = True
    results: Optional[List[SearchResultItem]] = None
    error: Optional[str] = None


# --- Orchestrator / General Task Schemas ---

class Task(BaseModel):
    id: str
    description: str
    agent_type: str # e.g., "db_query", "code_exec", "search"
    params: Dict[str, Any]
    status: str = "pending" # pending, in_progress, completed, failed
    result: Optional[Any] = None
    dependencies: List[str] = [] # List of task IDs this task depends on

class Plan(BaseModel):
    tasks: List[Task]
    execution_order: List[List[str]] # Groups of task IDs that can be run in parallel

class OrchestratorRequest(BaseModel):
    user_prompt: str

class OrchestratorResponse(BaseModel):
    plan: Optional[Plan] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
