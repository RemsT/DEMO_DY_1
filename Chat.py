import os
import config
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone as pinecone_init

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"]
index_name = st.secrets["INDEX_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

st.set_page_config(page_title="DY Chat")

st.sidebar.subheader('Documents:')
st.sidebar.write('- Advicary circular: Airport Terminal Planning')
st.sidebar.write('- ACRP synthesis 97: How Airports Plan for Changing Aircraft Capacity: The Effects of Upgauging')
st.sidebar.write('- ACRP Report 25: Airport Passenger Terminal Planning and Design')

# LOAD THE VECTOR DATABASE AND PREPARE RETRIEVAL
from langchain_pinecone import PineconeVectorStore

# Initializing Pinecone Vector DB
pc = pinecone_init( api_key=pinecone_api_key)
index = pc.Index("aviation")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

llm = ChatOpenAI( model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.2)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory()

# CONTEXT PROMPT
### Contextualize question ###
contextualize_q_assistant_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, if you are not totally sure."
    "If you  are missing information you can respond that you don't have enough information."
    "when number are part or the response please make sure you have the right value."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("assistant", contextualize_q_assistant_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# PREPARE PROMPT
assistant_prompt = (
    "Your task is to answer clients questions as truthfully as possible."
    "Use the provided retrieved information  "
    "help answer the questions. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "If you are using number make sure you are using the correct values."
    "If you  are missing information you can respond that you don't have enough information."
    "If you are not sure you can say that you are not sure"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("assistant", assistant_prompt), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# TITLE
st.markdown("<h1 style='text-align: center;'>Demo DY consultant</h1> <br>", unsafe_allow_html=True)

st.info(
    "Chat with documents"
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if input := st.chat_input():
    st.chat_message("human").write(input)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "abc123"}}
    response = conversational_rag_chain.invoke({"input": input}, config)
    
    with st.chat_message("assistant"):
        st.write(response["answer"])
        with st.expander("See 3 first sources:"):
            for doc in response["context"]:
                source = os.path.split(doc.metadata["source"])[1] + "--->   Page: "+ str(doc.metadata["page"])
                st.write(source)
