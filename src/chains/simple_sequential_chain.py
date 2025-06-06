from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


llm = ChatOllama(model="llama3.2:3b") #llama 3.2 with 3 billion params

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.
    You need to craft an impactful title for a speech on the following topic: {topic}
    Answer exactly with one title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    You need to write a powerful speech of 350 words for the following title: {title}
    """
)


st.title("Ask me to write a speech on any topic!")

first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])
second_chain = speech_prompt | llm
final_chain = first_chain | second_chain

topic = st.text_input("Enter the topic for the speech")

if topic:
    response = final_chain.invoke({
        "topic":topic
    })

    st.write(response.content)