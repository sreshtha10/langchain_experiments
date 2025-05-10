with open('../api_key.txt','r') as file:
    apiKey = file.read().strip()


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o",api_key=apiKey)

question = input('Enter the question!')
response = llm.invoke(question)

print(response)

