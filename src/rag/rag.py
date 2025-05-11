from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b")

document = TextLoader(file_path="src/utils/product-data.txt").load()

text_splitters = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
chunks = text_splitters.split_documents(documents=document)
vector_store = Chroma.from_documents(chunks,embedding)

retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an assitant for answering questions. Use the provided context to respond. If the answer isn't clear, acknowledge that you don't know. Limit your response to three concise sentences. {context}"),
        ("human","{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm=llm,prompt=prompt_template)
rag_chain = create_retrieval_chain(retriever,qa_chain)

print('Chat with Document\n')

question = input("Enter your question\n")

if question:
    response = rag_chain.invoke(
        {"input":question}
    )
    print(response['answer'])