import os
import config
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = st.secrets["INDEX_KEY"]

# Initializing Pinecone Vector DB
pinecone.init(
    api_key=pinecone_api_key,
    environment=PINECONE_ENV
)

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
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma.db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)

st.write("Vector database created with your documents")
