from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
import streamlit as st



llm = ChatOllama(model="mistral:latest")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an agile coach. Answer any questions related to the agile process"),
        ("human","{input}")
    ]
)

st.title("Ask Our Trip Planner!")

user_input = st.text_input("Enter your query")

chain = prompt | llm

if user_input:
    response = chain.invoke({"input":user_input})
    st.write(response.content)