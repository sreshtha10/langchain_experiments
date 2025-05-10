from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


llm = ChatOllama(model="llama3.2:3b")

prompt_template= ChatPromptTemplate.from_messages(
    [
        ("system","You are an agile coach. Answer the questions asked related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)

st.title("Agile Guide")
user_input = st.text_input("Enter your question")

chain = prompt_template | llm
history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if user_input:
    response = chain_with_history.invoke(
        input={"input":user_input},
        config={"configurable":{
            "session_id":"abc123"
        }}
    )

    st.write(response.content)


st.write("HISTORY")
st.write(history_for_chain)