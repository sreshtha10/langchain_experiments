from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st

llm = ChatOllama(model="gemma:2b")
st.title("Ask Gemma:2b a Question!")

promp_template = PromptTemplate(
    input_variables=["country"],
    template="""
    You are an expert on food. Answer the following question: What is the most famous dish
    of the {country}?
    """
)

question = st.text_input("Enter the country?")

if question:
    response = llm.invoke(promp_template.format(country=question))
    st.write(response.content)