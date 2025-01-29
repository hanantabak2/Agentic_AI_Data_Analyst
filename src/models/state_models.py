from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Task(BaseModel):
    """
    Defines a task structure with step-by-step details and output dictionary specifications.

    Structure to Follow (Step-by-Step):
    -------------------
    task-1: Precise action description.
    column_name: "column-1"..

    task-2: Precise action description.
    column_name: "column-1"..

    task-N: Compile the processed results and store them in the final output dictionary named `output_dict`.
    - key_names: ["Key-1", "Key-2", ...]
    - values: [
        "Key-1"-"Description of the information contained in this key",
        "Key-2"-"Description of the information contained in this key",
        ...
    ]
    """
    task: str = Field(None, description="Step-by-step detailed task description for the user query that need to be performed.")
    column_name: str = Field(None, description="Provide all the column names used for Task, formatted as a comma-separated.")
    key_names: List[str] = Field(None, description="List of keys to be used in the final output dictionary")
    values: List[str] = Field(None,description="List containing the final results of the task plan with Key-Value pairs.")

class Tasks(BaseModel):
    tasks: List[Task] = Field(
        description="List of tasks to be performed to answer the user query.",
    )

class Code(BaseModel):
    final_code: str = Field(
        description="Final Python code generated to perform the task plan for solving the user query.",
    )

class State(TypedDict):
    task_plan: list[Task]
    user_query: str
    code: str
    output: dict
    error: str
    iterations: int
    answer: str