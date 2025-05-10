from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


llm = OllamaEmbeddings(model="nomic-embed-text")

document = TextLoader("src/utils/job_listings.txt",encoding="utf-8").load()

text_splitters = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)

chunks = text_splitters.split_documents(documents=document)

db = Chroma.from_documents(chunks,llm)

user_input = input("Enter the query")
embedding_vector = llm.embed_query(user_input)

docs = db.similarity_search_by_vector(embedding_vector)

for doc in docs:
    print(doc.page_content)
