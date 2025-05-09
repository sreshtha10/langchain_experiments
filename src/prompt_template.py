from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st

llm = ChatOllama(model="gemma:2b")
st.title("Ask Gemma:2b a Question!")

promp_template = PromptTemplate(
    input_variables=["country","number_of_dishes","response_language"],
    template="""
    You are an expert on food. Answer the following question: What is the top {number_of_dishes} famous dish
    of the {country} in the language {response_language}
    Avoid giving information about fictional places. If the country is fictional then simply say I don't know.
    """
)

country = st.text_input("Enter the country")
number_of_dishes = st.number_input("Enter the number of dishes",min_value=1,max_value=20)
response_language= st.text_input("Enter the response language")

if country:
    response = llm.invoke(promp_template.format(country=country,number_of_dishes=number_of_dishes,response_language=response_language))
    st.write(response.content)