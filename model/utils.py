import streamlit as st
from llama_index import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter

TEXT_CARET = "\u258c"
PAT = st.secrets.CLARIFAI_PAT
ENV = st.secrets.ENV
VERBOSE = True if ENV == "dev" else False


@st.cache_resource()
def get_document():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_langchain_documents()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    return documents
