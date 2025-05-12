from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64
import streamlit as st



llm = ChatOllama(model="llama3.2:3b")

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant that can describe an image based on the input text"),
        ("human",[
            {"type":"text", "text":"{input}"},
            {
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/jpeg;base64,""{image}",
                    "detail":"low"
                }
            }
        ])
    ]
)


chain = prompt_template | llm

uploaded_file = st.file_uploader("Upload your image",type=["jpg","jpeg","png"])
question = st.text_input("Enter a question")


if question:
    image = encode_image(uploaded_file)
    response = chain.invoke({
        "input":"Explain",
        "image":image
    })

    st.write(response.content)


