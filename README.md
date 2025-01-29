# Data Analysis Assistant

An AI-powered data analysis application that helps users analyze CSV/EXCEL data through natural language questions. Built with Streamlit, LangChain,Langgraph, and GPT-4o-mini.

## Features

- Upload and analyze CSV/EXCEL files
- Ask questions about your data in natural language
- Get automated data analysis and visualizations
- Download processed data as CSV
- View detailed analysis plans and generated code
- Interactive visualizations using Plotly

## Setup

1. Clone the repository:
```bash
git clone [<repository-url>](https://github.com/JoshiSneh/Data_Visualization_Python_Langgraph.git)
cd streamlit-app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Upload your CSV/EXCEL file using the file uploader
2. View a preview of your data
3. Ask questions about your data in natural language
4. Explore the results, visualizations, and download processed data

## Dependencies

- langchain-openai
- langchain-community
- langchain-core
- langgraph
- pandas
- plotly
- python-dotenv
- pydantic
- streamlit
- typing-extensions

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.
