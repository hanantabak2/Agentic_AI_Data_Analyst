import streamlit as st
import pandas as pd
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
import plotly.graph_objects as go
import time
from src.graph.workflow import create_workflow

st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()
    
    st.subheader("OpenAI API Settings")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    if api_key:
        if not api_key.startswith(('sk-', 'org-')):
            st.error("Invalid API key format")
            st.stop()
        st.session_state.openai_api_key = api_key
        st.success("API key configured! ✅")
    else:
        st.warning("⚠️ Please enter your OpenAI API key")
        st.stop()


st.title("📊 Data Analysis Assistant")

tab1, tab2 = st.tabs(["🎯 Overview", "📖 Instructions"])

with tab1:
    st.markdown("""
    Welcome to the Data Analysis Assistant! This tool helps you:
    - 📈 Analyze complex datasets with natural language
    - 🎨 Generate insightful visualizations
    - 📝 Create detailed summaries and reports
    """)

with tab2:
    st.markdown("""
    **How to use:**
    1. Upload your CSV or Excel file
    2. Preview your data to ensure it's loaded correctly
    3. Ask questions in natural language
    4. Watch as the assistant analyzes your data step by step
    """)

uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    try:
        file_size = uploaded_file.size
        if file_size > 200 * 1024 * 1024:
            st.error("File size exceeds 200MB limit")
            st.stop()
        
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if df.empty:
            st.error("The file contains no data")
            st.stop()
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            top_p=0.2,
            api_key=st.session_state.openai_api_key
        )
        workflow = create_workflow(llm, df)

        st.dataframe(
            df,
            use_container_width=True,
            height=300
        )
        
        st.divider()
        st.subheader("🔍Ask Questions About Your Data")
        
        question = st.text_input(
            "",
            placeholder="e.g., 'Show monthly sales trends' or 'Create a summary of revenue by region'",
            key="question_input"
        )
        
        if st.button("🚀 Analyze", use_container_width=True):
            if not question:
                st.warning("Please enter a question first")
                st.stop()

            try:
                with st.status(label="🤖 Analysis in Progress", state="running") as status:
                    with get_openai_callback() as cb:
                        
                        state = workflow.invoke({
                            "user_query": question,
                            "error": None,
                            "iterations": 0
                        })
                        
                        st.subheader("📋 Analysis Plan")
                        st.code(state["task_plan"], language='python')
  
                        st.subheader("🔍 Generated Code")
                        st.code(state["code"], language='python')
                        
                        if state.get("output"):
                            st.subheader("📈 Analysis Results")
                            
                            for key, value in state["output"].items():
                                if isinstance(value, pd.DataFrame):
                                    st.write(f"📊 {key}")
                                    st.dataframe(value, use_container_width=True)
                                elif isinstance(value, go.Figure):
                                    st.plotly_chart(value, use_container_width=True)
                                elif isinstance(value, pd.Series):
                                    st.write(f"📊 {key}")
                                    st.dataframe(pd.DataFrame(value), use_container_width=True)
                                else:
                                    st.info(f"{key}: {value}")                   
                        time.sleep(2.0)  
                        
                        if state.get("answer"):
                            st.subheader("📌 Summary")
                            st.success(state["answer"])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Total Tokens Used: {cb.total_tokens}")
                        with col2:
                            st.caption(f"Analysis Cost: ${cb.total_cost:.4f}")
                status.update(label="✅ Analysis Complete", state="complete")


            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("💡 Try rephrasing your question or check if your data contains the requested information.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("💡 Please ensure your file is properly formatted and try again.")
else:
    st.info("👆 Upload a CSV or Excel file to begin your analysis!")
