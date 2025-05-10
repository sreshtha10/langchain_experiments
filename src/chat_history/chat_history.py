from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory



llm = ChatOllama(model="llama3.2:3b")

prompt_template = ChatPromptTemplate(
    [
        ("system","You are an agile coach. Answer any questions related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)

chain = prompt_template | llm

history_for_chain = ChatMessageHistory()


chat_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)


print("Agile Guide")

while True:
    question = input("Enter the question!")
    if question:
        response = (chat_with_history.invoke(
            {"input":question},
            {"configurable":{
                "session_id":"abc123"
            }}
        ))

        print(response)

