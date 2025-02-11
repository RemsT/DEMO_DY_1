import os
import config
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as pinecone_init, ServerlessSpec

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"]
index_name = st.secrets["INDEX_KEY"]

# Initializing Pinecone Vector DB
pc = pinecone_init( api_key=pinecone_api_key)
index = pc.Index("aviation")

st.title("Upload pdf files and create your database")

uploaded_files = st.file_uploader(label="Upload PDF files", type=["pdf"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# Read documents
docs = []
temp_dir = tempfile.TemporaryDirectory()
for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# SAVE TO DISK
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

vectorstore_from_docs = PineconeVectorStore.from_documents(
        splits,
        index_name=index_name,
        embedding=embeddings
        )

st.write("Vector database created with your documents")
