from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity

embedding = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    Document(page_content="Apple"),
    Document(page_content="Orange"),
    Document(page_content="BMW")
]


vector_store = []

for doc in docs:
    vector = embedding.embed_query(doc.page_content)
    vector_store.append(vector)



user_input = input("Enter the text!\n")

query_vector = embedding.embed_query(user_input)

similarity = cosine_similarity([query_vector],vector_store)[0]

print(similarity)