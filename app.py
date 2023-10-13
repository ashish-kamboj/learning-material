from src.constants import aliases
from pathlib import Path
import streamlit as st
import os

# Define a function to read the markdown files
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text(encoding='utf-8')

# Get the list of markdown files in the current directory
files = list(Path("files/").glob("*.md"))

# Define a function that returns the alias for each file name
def get_alias(file_name):
    file_name_str = str(os.path.basename(file_name))  # Convert WindowsPath object to string
    return aliases.get(file_name_str, file_name_str)

# Create a drop down list of file names with aliases
file = st.sidebar.selectbox("Select a markdown file", files, format_func=get_alias)

# Display the content of the selected file
content = read_markdown_file(file)
st.markdown(content, unsafe_allow_html=True)