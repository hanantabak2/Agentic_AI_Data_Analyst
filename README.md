# Data Analysis Project

This data analysis application helps users analyze CSV/EXCEL data through natural language questions allowing users to questions about a dataset and receive insightful visual representations in response.


![Demo](https://github.com/user-attachments/assets/ea551c6d-2e75-4fac-82a2-683c8ddcf6c5)
- Working Demo:

## Workflow

![workflow](https://github.com/user-attachments/assets/3e990c3d-63e0-4e69-93c2-95a29ffd5fb8)

## Features

- Upload and analyze CSV/EXCEL files
- Ask questions about your data in natural language
- Get automated data analysis and visualizations
- View detailed analysis plans and generated code
- Interactive visualizations using Plotly

## Setup

1. Clone the repository:
```bash
git clone https://github.com/JoshiSneh/Data_Visualization_Python_Langgraph.git
cd Data_Visualization_Python_Langgraph
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
4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Upload your CSV/EXCEL file using the file uploader
2. View a preview of your data
3. Ask questions about your data in natural language
4. Explore the results, visualizations

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
