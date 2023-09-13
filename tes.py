
import streamlit as st
import random
import time
from langchain.llms import Clarifai
from llama_index import SimpleDirectoryReader
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.embeddings import ClarifaiEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS, Weaviate
import weaviate

reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
documents = reader.load_langchain_documents()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = ClarifaiEmbeddings(
    pat="87c464ced9cc4d9ca739a1e6aac22772",
    user_id="openai",
    app_id="embed",
    model_id="text-embedding-ada"
)
vectorstore = FAISS.from_documents(documents, embeddings)

vectorstore.save_local("vectorstore2")
