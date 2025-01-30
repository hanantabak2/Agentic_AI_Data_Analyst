import streamlit as st
import pandas as pd
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
import plotly.graph_objects as go

from src.graph.workflow import create_workflow

# load_dotenv()

st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📊",
    layout="centered",
)

with st.sidebar:
    st.subheader("OpenAI API Key")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        try:
            if not api_key.startswith(('sk-', 'org-')):
                st.error("Invalid API key format. Please check your API key.")
                st.stop()
            st.session_state.openai_api_key = api_key
        except Exception as e:
            st.error(f"Error validating API key: {str(e)}")
            st.stop()
    else:
        st.error("⚠️ Please enter your OpenAI API key to proceed")
        st.stop()

if "openai_api_key" not in st.session_state:
    st.stop()

st.markdown("""
    <style>
        .stAlert {
            margin-top: 1rem;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .st-emotion-cache-16idsys p {
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
        }
        .error-message {
            color: #ff4b4b;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #ffe5e5;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Data Analysis Assistant")
st.markdown("""
    Upload your CSV/EXCEL file and ask questions about your data. The assistant will analyze 
    your data and provide insights through visualizations and summaries.
""")

uploaded_file = st.file_uploader("Choose a CSV/EXCEL file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        file_size = uploaded_file.size
        if file_size > 200 * 1024 * 1024: 
            st.error("File size too large. Please upload a file smaller than 200MB.")
            st.stop()

        file_type = uploaded_file.type
        if file_type == "text/csv":
            try:
                df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty. Please check the file content.")
                st.stop()
            except pd.errors.ParserError:
                st.error("Unable to parse the CSV file. Please ensure it's properly formatted.")
                st.stop()
        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                st.stop()
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop()

        if df.empty:
            st.error("The uploaded file contains no data. Please check the file content.")
            st.stop()
        
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                top_p=0.2,
                api_key=st.session_state.openai_api_key
            )
            worflow = create_workflow(llm, df)
        except Exception as e:
            st.error(f"Error initializing AI model: {str(e)}")
            st.info("This might be due to an invalid API key or connection issues.")
            st.stop()
        
        with st.expander("Preview Data"):
            st.dataframe(df, use_container_width=True)
            st.caption(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
        
        question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., Show me monthly trends, Create a summary of sales, etc."
        )
        
        if st.button("🚀 Analyze Data", use_container_width=True):
            if not question:
                st.error("Please enter a question to analyze your data.")
                st.stop()
            
            try:
                with st.spinner("Analyzing your data..."):
                    # config = {"configurable": {"thread_id": "1"}}
                    with get_openai_callback() as cb:
                        state = worflow.invoke({
                            "user_query": question,
                            "error": None,
                            "iterations": 0
                        })
                    
                    st.subheader("Analysis Plan")
                    print(state["task_plan"])
                    st.code(state["task_plan"], language='python')
                    # for index, task in enumerate(state["task_plan"],1):
                    #     st.markdown(f"**Step {index}**: {task.task}")
                    #     if index == len(state["task_plan"])-1:
                    #             st.write(f"{task.key_names}")
                    #             for value in task.values:
                    #                 st.write(f"{value}")
                    
                    with st.expander("View Generated Code"):
                        st.code(state["code"], language='python')
                    
                    if state.get("output"):
                        st.subheader("Results")
                        
                        for key, value in state["output"].items():
                            if isinstance(value, pd.DataFrame):
                                st.write(f"📈 {key}")
                                st.dataframe(value, use_container_width=True)
                                buffer = io.StringIO()
                                value.to_csv(buffer, index=False)
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=buffer.getvalue(),
                                    file_name=f"{key.lower().replace(' ', '_')}.csv",
                                    mime="text/csv"
                                )
                            elif isinstance(value, go.Figure):
                                st.plotly_chart(value, use_container_width=True)
                            else:
                                st.info(f"{key}: {value}")
                    else:
                        st.warning("No output was generated for this analysis. This might be due to insufficient data or an unclear question. Please try rephrasing your question.")
                    
                    if state.get("error"):
                        st.error(f"Error during analysis: {state['error']}")
                        st.info("Try rephrasing your question or check if your data contains the requested information.")
                    
                    if state.get("answer"):
                        st.subheader("Summary")
                        st.success(state["answer"])
                    
                    st.caption(f"Total Tokens: {cb.total_tokens}")
                    st.caption(f"Total Cost: ${cb.total_cost:.4f}")
            
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {str(e)}")
                st.info("Please try again with a different question or check your data format.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file is properly formatted and try uploading again.")
else:
    st.info("👆 Please upload a CSV/EXCEL file to begin analysis.")















    

# from typing import Annotated, List, Optional
# from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# load_dotenv()
# import pandas as pd
# import plotly.graph_objects as go
# import io
# import plotly.express as px
# import numpy as np
# from datetime import datetime as dt
# from langchain_community.callbacks.manager import get_openai_callback
# import re

# from pydantic import BaseModel,Field
# from langgraph.graph import StateGraph, START, END
# from IPython.display import display, Image
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_deepseek import ChatDeepSeek
# import streamlit as st

# df = pd.read_csv(r"C:\Personal\Machine Learning\GenAI - Langchain\Idea Elan\Adhoc_Report\Custom_Reports\Transaction_Custom_Report.csv")

# data_frame_preview = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.items()])
# available_columns = ', '.join(df.columns)
# column_data_types = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.dtypes.items()])

# ### Task Planning
# class Task(BaseModel):
#     """
#     Defines a task structure with step-by-step details and output dictionary specifications.

#     Structure to Follow (Step-by-Step):
#     -------------------
#     task-1: Precise action description.
#     column_name: "column-1"..

#     task-2: Precise action description.
#     column_name: "column-1"..

#     task-N: Compile the processed results and store them in the final output dictionary named `output_dict`.
#     - key_names: ["Key-1", "Key-2", ...]
#     - values: {
#         "Key-1": "Description of the information contained in this key",
#         "Key-2": "Description of the information contained in this key",
#         ...
#     }
#     """

#     task: str = Field(None, description="Step-by-step detailed task description for the user query that need to be performed.")
#     column_name: str = Field(None, description="Provide all the column names used for Task, formatted as a comma-separated.")
#     key_names: List[str] = Field(None, description="List of keys to be used in the final output dictionary.")
#     values: List[str] = Field(None,description="List containing the final results of the task plan with Key-Value pairs.")

# class Tasks(BaseModel):
#     tasks: List[Task] = Field(
#         description="List of tasks to be performed to answer the user query.",
#     )


# ### Task Execution

# class Code(BaseModel):
#     final_code: str = Field(
#         description="Final Python code generated to perform the task plan for solving the user query.",
#     )


# llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,top_p=0.2)
# # llm = ChatDeepSeek(model="deepseek-chat",temperature=0,top_p=0.2)


# class State(TypedDict):
#     task_plan:list[Task]
#     user_query:str
#     code:str
#     output:dict
#     error:str
#     iterations:int
#     answer:str


# def plan_task(state: State) -> dict:

#     task_planner_prompt = """
#             ### Objective
#             You are an expert task planning agent. Your role is to create precise, executable task plans for analyzing the DataFrame 'df'. Think step-by-step and generate a detailed task plan.

#             ### Task Planning Guidelines
#             - Make sure each task is related to the user query and provides a clear and correct action description that can be converted to the Python Code.
#             - Use the exact column names as specified in the [Available Columns].
#             - Think step-by-step and provide a detailed plan for each task that needs to be performed to answer the user query.
#             - Last Final Task should compile the processed results and store them in the final output dictionary named 'output_dict'.
#             - Reset indexs before storing DataFrames in the 'output_dict'.

#             ### Graph Requirements
#             - Use only the 'plotly' library for creating graphs.
#             - Recommend the most suitable graph type for the given user query based on the DataFrame's data.

#             ### Output Guidelines
#             - Store the final output in a dictionary named 'output_dict' containing all results such as dataframes, variables, and graphs.
#             - Ensure keys in 'output_dict' are formatted with the first word capitalized and space-separated words.
#             """

#     prompt = ChatPromptTemplate.from_messages([
#         ("system",task_planner_prompt),
#         ("human","===Dataframe Schema:\n{data_frame_preview}\n\n===Available Columns:\n{available_columns}\n\n===Column Data Types:\n{column_data_types}\n\n===User Question:\n{user_question}\n")
#     ])
    
#     task_planner_llm = llm.with_structured_output(Tasks)

#     task_chain = prompt | task_planner_llm

#     response = task_chain.invoke({"data_frame_preview":data_frame_preview, "available_columns":available_columns, "column_data_types":column_data_types, "user_question":state["user_query"]})

#     return {"task_plan":response.tasks}

# def execute_task(state: State) -> dict:

#     error = state["error"]

#     if error:
#         retry_llm = llm.with_structured_output(Code)
#         code = state["code"]
#         task_plan = state["task_plan"]
#         response = retry_llm.invoke(
#             f"You have been given error: {error} and Python code: {code} along with Task Plan {task_plan}. Please fix the error and generate the correct python code to perform the task plan. Donot change the task plan and the column names. Use the existing 'df' and strictly follow the given plan.")
        
#         iterations = state["iterations"]

#         print(f"Attempt #{iterations + 1}")
#         return {"code":response.final_code, "iterations": iterations + 1}
#     else:
#         python_code_prompt = """
#             ### Objective
#             You are an expert data analysis assistant with advanced knowledge of pandas, numpy, and plotly. Respond with code only, using the existing 'df' and strictly following the given plan. You have knowledge of the previous error and should generate correct complex Python code for data manipulation and visualization.

#             ### Data Operations
#             - Do not recreate 'df' or assume additional data.
#             - Dataframe 'df' is already loaded and available for use.
#             - If the operation can be performed without regex, avoid using it.
#             - Do not assume any additional data; use only from the existing [Dataframe Schema].
#             - Use exact column names as specified in [Execution Plan].
#             - Use exact column types for the code generation as specified in [Column Data Types].
#             - Preserve data types; avoid filling nulls arbitrarily.
#             - Use descriptive variable names for intermediate DataFrames.
#             - Convert date, month, year columns to datetime objects where necessary.
#             - Do not convert any of the DataFrames to a list(.tolist()) or dictionary(.to_dict()) for the result dataframes.Keep them as DataFrames only. Result dataframes are those that are stored in the 'output_dict' dictionary.

#             ### Code Standards
#             - Import all necessary libraries.
#             - All tasks should be executed in order step-by-step. It should never be like:# (This is already done in the previous step).
#             - Use up-to-date pandas methods.
#             - Maintain clear, consistent naming.
#             - Code should be correct and run on all Python environments and versions.
#             - Convert all lists to DataFrames before storing them in the 'output_dict'.
#             - Perform all the operations before storing them in the 'output_dict'. Last task should alwyas be compiling the results in the 'output_dict'.

#             ### Visualization Standards
#             - Use Plotly only.
#             - Provide sensible figure sizing, labeling, and coloring.
#             - Ensure interactive capabilities where beneficial.
#             - Do not use fig.show() or plt.show() for visualization.

#             ### Output Requirements
#             - Code only, no additional explanations or text.
#             - No print statements unless explicitly required.
#             - No markdown or commentary.
#             - Steps should be numbered according to the plan.
#             - Output should always be of a dictionary type named 'output_dict'.
#             """

#         prompt = ChatPromptTemplate.from_messages([
#             ("system",python_code_prompt),
#             ("human","===Dataframe Schema:\n{data_frame_preview}\n\n===Available Columns:\n{available_columns}\n\n===Column Data Types:\n{column_data_types}\n\n===Execution Plan:\n{execution_plan}\n\n===User Question:\n{user_question}\n\n")
#         ])

#         task_plan = state["task_plan"]
        
#         code_llm = llm.with_structured_output(Code)

#         code_chain = prompt | code_llm

#         response = code_chain.invoke({"data_frame_preview":data_frame_preview, "available_columns":available_columns, "column_data_types":column_data_types,"execution_plan":task_plan, "user_question":state["user_query"],"error":state["error"]})

#         iterations = state["iterations"]

#         print(f"Attempt #{iterations + 1}")
#         return {"code":response.final_code, "iterations": iterations + 1}


# def execute_with_exec(state: State) -> str:
#     """
#     Execute Python code string using exec()
#     """

#     try:
#         exec_globals = {"df":df, "pd": pd, "px": px, "np": np,"re":re,"dt":dt,"go":go}
#         exec_locals = {}
        
#         code = state["code"]
#         iterations = state["iterations"]

#         exec(code, exec_globals, exec_locals)
        
#         if "output_dict" not in exec_locals:
#             raise ValueError("Missing output_dict")
        
#         print(f"Success after attempt #{iterations}")
#         return {"output":exec_locals["output_dict"], "error":None}
        
#     except Exception as e:
#         print(f"Failed attempt #{iterations} with error: {str(e)}")
#         return {"code":code, "error":str(e)}
    
# def retry_code(state: State) -> str:
#     """Determine if we should retry on error"""
#     error = state["error"]
#     iterations = state["iterations"]

#     if error == None:
#         return "END"
#     if iterations < 3:
#         print(f"Retrying after attempt #{iterations}")
#         return "RETRY"
#     print(f"Giving up after {iterations} attempts")
#     state["output"] = None
#     return "STOP"

# def format_result(state: State) -> dict:

#     format_result_prompt = """
#         You are an AI assistant that formats python results into a human-readable response like a summary. Give a conclusion to the user's question based on the python results. Do not give the answer in markdown format.
#         """

#     prompt = ChatPromptTemplate.from_messages([
#         ("system",format_result_prompt),
#         ("human","===User Question:\n{user_question}\n\n===Python Results:\n{result}\n\nFormatted response:"),
#     ])
    
#     format_llm = llm

#     format_chain = prompt | format_llm

#     response = format_chain.invoke({"user_question":state["user_query"],"result":state["output"]})

#     return {"answer":response.content}

# workflow = StateGraph(State)

# workflow.add_node("Plan Task", plan_task)
# workflow.add_node("Execute Task", execute_task)
# workflow.add_node("Code Execution", execute_with_exec)
# workflow.add_node("Format Result", format_result)

# workflow.add_edge(START, "Plan Task")
# workflow.add_edge("Plan Task", "Execute Task")
# workflow.add_edge("Execute Task", "Code Execution")
# workflow.add_conditional_edges("Code Execution", retry_code, {"RETRY": "Execute Task", "STOP":END,"END": "Format Result"})
# workflow.add_edge("Format Result", END)

# chain = workflow.compile()

# image_data = chain.get_graph().draw_mermaid_png()
# question = st.text_input("Enter your question: ")

# if question:
#     with get_openai_callback() as cb:
#         state = chain.invoke({"user_query": question,"error":None, "iterations": 0})

#     st.header("Task Plan")
#     # st.write(state["task_plan"])

#     for index,task in enumerate(state["task_plan"]):
#         st.write(f"📋 {task.task}")
#         # if index == len(state["task_plan"])-1:
#         #     st.write(f"{task.key_names}")
#             # for key, value in task.values.items():
#             #     st.write(f"{key}: {value}")
        

#     st.header("Generated Code")
#     st.code(state["code"], language='python')

#     if state["output"]:
#         st.header("Output")
#         for key, value in state["output"].items():
#             if isinstance(value, pd.DataFrame):
#                     st.write(f"📈 {key}")
#                     st.dataframe(value, use_container_width=True)

#                     buffer = io.StringIO()
#                     value.to_csv(buffer, index=False)
#                     st.download_button(
#                         label="📥 Download as CSV",
#                         data=buffer.getvalue(),
#                         file_name=f"{key.lower().replace(' ', '_')}.csv",
#                         mime="text/csv"
#                     )
#             elif isinstance(value, go.Figure):
#                 st.plotly_chart(value, use_container_width=True)
#                 # graph_visual[key] = value.to_json()
#                 visual = True
#             else:
#                 st.warning(f"{key}: {value}")
    
#     if state["error"]:
#         st.error(state["error"])
    
#     st.header("Summary")
#     st.success(state["answer"])

#     st.header("Cost")
#     st.caption(f"Total Tokens: {cb.total_tokens}")
#     st.caption(f"Total Cost: ${cb.total_cost}")
    
#     st.image(image_data,width=200)


