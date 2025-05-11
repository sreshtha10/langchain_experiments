from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b")

document = TextLoader(file_path="src/utils/product-data.txt").load()

text_splitters = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
chunks = text_splitters.split_documents(documents=document)
vector_store = Chroma.from_documents(chunks,embedding)

retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an assitant for answering questions. Use the provided context to respond. If the answer isn't clear, acknowledge that you don't know. Limit your response to three concise sentences. {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)


history_aware_retriever = create_history_aware_retriever(llm,retriever,prompt_template)
qa_chain = create_stuff_documents_chain(llm=llm,prompt=prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history= RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)


st.title("Hi")
user_input = st.text_input("Enter your question")

if user_input:
    response = chain_with_history.invoke(
        {"input":user_input},
        {"configurable":{
            "session_id":"abc12",
        }}
    )

    st.write(response)


st.write("History")
st.write(history_for_chain)