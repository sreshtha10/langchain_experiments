from langchain_community.chat_models import ChatOllama
import streamlit as st

llm = ChatOllama(model="gemma:2b")

st.title('Ask a Question!')

question = st.text_input('')

if question:
    response = llm.invoke(question)
    st.write(response.content)