from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS


embedding = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the captial of Germany")
]

for i, doc in enumerate(docs, 1):
    vector = embedding.embed_query(doc.page_content)
    print(f"\nDocument {i}: {doc.page_content}")
    print(f"Vector ({len(vector)} dimensions):\n{vector}")