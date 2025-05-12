# Experimenting with the following topics related to LLM and Langchains:

1. Using OpenAI via Langchain
2. Using Langchain community to use OpenSource Models.
3. Prompt Templates
4. LCEL and Chains
5. Mainting Chat History using ChatPromptTemplates
6. Embeddings and Similarity Finder
7. Vector Stores and Retrievers
8. RAG - Retrieval Augmented Generation
9. Image Processing
10. Agents
    

# 🔍 LLM & LangChain Exploration

This project focuses on hands-on experimentation with **LangChain** and **LLMs** (Large Language Models), both from **OpenAI** and open-source providers. The aim is to understand and build core functionalities such as prompt engineering, chains, RAG, embeddings, and agent-based systems.

## 1. 🧠 Using OpenAI via LangChain
Integrating OpenAI models (like GPT-4) into LangChain workflows using wrappers like `ChatOpenAI`, enabling interaction with powerful LLMs via a simple interface. This provides better control over prompts, memory, tools, and more.

## 2. 🌐 Using LangChain Community to Use Open-Source Models
Utilizing the `langchain_community` module to access and connect with local or hosted open-source models (e.g., Llama, Mistral) through HuggingFace APIs or local endpoints. Offers flexibility beyond proprietary LLMs.

## 3. ✍️ Prompt Templates
`PromptTemplate` and `ChatPromptTemplate` allow reusable and structured prompts. They enable modular prompt engineering with placeholders that can be dynamically filled during execution, reducing repetition and errors.

```python
prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
```

## 4. 🔗 LCEL and Chains
**LangChain Expression Language (LCEL)** provides a declarative way to compose chains:
- Chains are sequences of operations (prompt → model → output parsing).
- LCEL simplifies building and reusing such flows.

```python
chain = prompt | llm | output_parser
```

## 5. 💬 Maintaining Chat History using `ChatPromptTemplates`
Useful for multi-turn conversations. Chat history is managed using `ChatMessageHistory` or memory components like `ConversationBufferMemory`. These preserve past user/AI messages in `ChatPromptTemplate`.

## 6. 📐 Embeddings and Similarity Finder
Embeddings are numerical representations of text. You can use models like `OpenAIEmbeddings` or `HuggingFaceEmbeddings` to convert text to vectors. Similarity search helps in finding the closest match from a dataset based on vector distance.

```python
from langchain.embeddings import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings()
```

## 7. 🧱 Vector Stores and Retrievers
Vector stores (e.g., FAISS, Chroma) store and index embeddings. Retrievers are used to fetch the most relevant documents during a query using semantic similarity.

```python
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()
```

## 8. 🔄 RAG - Retrieval Augmented Generation
RAG enhances LLM responses by injecting context from retrieved documents before generating a reply. Combines retrievers + LLMs to build knowledge-aware chatbots or Q&A systems.

```python
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

## 9. 🖼️ Image Processing
LangChain supports multimodal tasks by integrating tools like **OpenAI's Vision API**, **Transformers for OCR**, or **custom image parsing pipelines**. Useful for analyzing, captioning, or extracting data from images.

```python
from langchain.output_parsers import JsonOutputParser
# With OpenAI Vision models via LangChain tools
```

## 10. 🧠 Agents
Agents use reasoning to decide what tools (functions, APIs, retrievers) to call step-by-step. LangChain agents can plan, act, and observe using tools like calculators, retrievers, or external APIs.

```python
from langchain.agents import initialize_agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```
