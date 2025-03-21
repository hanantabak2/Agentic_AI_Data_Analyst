from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Code(BaseModel):
    final_code: str = Field(
        description="Final Python code generated to perform the task plan for solving the user query.",
    )

class State(TypedDict):
    task_plan: str
    user_query: str
    code: str
    output: dict
    error: str
    iterations: int
    answer: str