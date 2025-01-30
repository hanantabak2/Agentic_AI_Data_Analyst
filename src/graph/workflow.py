from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()

from src.models.state_models import Code, State

from src.prompts.system_prompts import (
    TASK_PLANNER_PROMPT,
    PYTHON_CODE_PROMPT,
    FORMAT_RESULT_PROMPT
)

def create_workflow(llm, df):
    def plan_task(state: State) -> dict:
        data_frame_preview = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.items()])
        available_columns = ', '.join(df.columns)
        column_data_types = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.dtypes.items()])

        prompt = ChatPromptTemplate.from_messages([
            ("system", TASK_PLANNER_PROMPT),
            ("human", "===Dataframe Schema:\n{data_frame_preview}\n\n===Available Columns:\n{available_columns}\n\n===Column Data Types:\n{column_data_types}\n\n===User Question:\n{user_question}\n")
        ])

        task_planner_llm = llm
        task_chain = prompt | task_planner_llm
        response = task_chain.invoke({
            "data_frame_preview": data_frame_preview,
            "available_columns": available_columns,
            "column_data_types": column_data_types,
            "user_question": state["user_query"]
        })

        return {"task_plan": response.content}

    def execute_task(state: State) -> dict:
        error = state["error"]

        if error:
            retry_llm = llm.with_structured_output(Code)
            code = state["code"]
            task_plan = state["task_plan"]
            response = retry_llm.invoke(
                f"You have been given error: {error} and Python code: {code} along with Task Plan {task_plan}. Please fix the error and generate the correct python code to perform the task plan. Donot change the task plan and the column names. Use the existing 'df' and strictly follow the given plan.")
            
            iterations = state["iterations"]
            print(f"Attempt #{iterations + 1}")
            return {"code": response.final_code, "iterations": iterations + 1}
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", PYTHON_CODE_PROMPT),
                ("human", "===Dataframe Schema:\n{data_frame_preview}\n\n===Available Columns:\n{available_columns}\n\n===Column Data Types:\n{column_data_types}\n\n===Execution Plan:\n{execution_plan}\n\n===User Question:\n{user_question}\n\n")
            ])

            data_frame_preview = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.items()])
            available_columns = ', '.join(df.columns)
            column_data_types = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.dtypes.items()])
            
            task_plan = state["task_plan"]
            code_llm = llm.with_structured_output(Code)
            code_chain = prompt | code_llm

            response = code_chain.invoke({
                "data_frame_preview": data_frame_preview,
                "available_columns": available_columns,
                "column_data_types": column_data_types,
                "execution_plan": task_plan,
                "user_question": state["user_query"],
                "error": state["error"]
            })

            iterations = state["iterations"]
            print(f"Attempt #{iterations + 1}")
            return {"code": response.final_code, "iterations": iterations + 1}

    def execute_with_exec(state: State) -> str:
        try:
            import pandas as pd
            import plotly.express as px
            import numpy as np
            import re
            from datetime import datetime as dt
            import plotly.graph_objects as go
            
            exec_globals = {
                "df": df,
                "pd": pd,
                "px": px,
                "np": np,
                "re": re,
                "dt": dt,
                "go": go
            }
            exec_locals = {}
            
            code = state["code"]
            iterations = state["iterations"]

            exec(code, exec_globals, exec_locals)
            
            if "output_dict" not in exec_locals:
                raise ValueError("Missing output_dict")
            
            print(f"Success after attempt #{iterations}")
            return {"output": exec_locals["output_dict"], "error": None}
            
        except Exception as e:
            print(f"Failed attempt #{iterations} with error: {str(e)}")
            return {"code": code, "error": str(e)}

    def retry_code(state: State) -> str:
        error = state["error"]
        iterations = state["iterations"]

        if error == None:
            return "END"
        if iterations < 3:
            print(f"Retrying after attempt #{iterations}")
            return "RETRY"
        print(f"Giving up after {iterations} attempts")
        state["output"] = None
        return "STOP"

    def format_result(state: State) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", FORMAT_RESULT_PROMPT),
            ("human", "===User Question:\n{user_question}\n\n===Python Results:\n{result}\n\nFormatted response:"),
        ])
        
        format_chain = prompt | llm
        response = format_chain.invoke({
            "user_question": state["user_query"],
            "result": state["output"]
        })

        return {"answer": response.content}


    workflow = StateGraph(State)


    workflow.add_node("Plan Task", plan_task)
    workflow.add_node("Execute Task", execute_task)
    workflow.add_node("Code Execution", execute_with_exec)
    workflow.add_node("Format Result", format_result)


    workflow.add_edge(START, "Plan Task")
    workflow.add_edge("Plan Task", "Execute Task")
    workflow.add_edge("Execute Task", "Code Execution")
    workflow.add_conditional_edges(
        "Code Execution",
        retry_code,
        {
            "RETRY": "Execute Task",
            "STOP": END,
            "END": "Format Result"
        }
    )
    workflow.add_edge("Format Result", END)

    return workflow.compile()