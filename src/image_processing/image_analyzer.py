from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64



llm = ChatOllama(model="llama3.2:3b")

def encode_image(image_path):
    with open(image_path,"rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    


image = encode_image("src/utils/airport_terminal_journey.jpeg")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant that can describe an image based on the input text"),
        ("human",[
            {"type":"text", "text":"{input}"},
            {
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/jpeg;base64,{image}",
                    "detail":"low"
                }
            }
        ])
    ]
)


chain = prompt_template | llm

response = chain.invoke({
    "input":"Explain"
})


print(response.content)