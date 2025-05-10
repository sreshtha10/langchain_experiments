from langchain_community.chat_models import ChatOllama
import streamlit as st
from langchain.globals import set_debug

set_debug(True)

llm = ChatOllama(model="gemma:2b")

st.title('Ask a Question!')

question = st.text_input('')

if question:
    response = llm.invoke(question)
    st.write(response.content)