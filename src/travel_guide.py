from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st



llm = ChatOllama(model="mistral:latest")

prompt = PromptTemplate(
    input_variables=["place","month","days"],
    template="""
    Create a travel itinerary for {place} for {days}. What are the must visit attractions during {month} and list some useful phrases in their local language.
    """
)

st.title("Ask Our Trip Planner!")

place = st.text_input("Enter the place you want to visit")
month = st.text_input("When are planning your trip?")
days = st.number_input("For how long are you visiting?", min_value=1)


if place and month and days:
    response = llm.invoke(prompt.format(place=place, month=month, days=days))
    st.write(response.content)