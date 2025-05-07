from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="gemma:2b")

question = input('Enter the question\n')

response = llm.invoke(question)

print(response)