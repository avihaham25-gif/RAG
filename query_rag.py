from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="qwen2.5:7b")
db = Chroma(persist_directory="./db", embedding_function=embeddings)

question = "מהם הטיעונים המרכזיים בנושא מערכת המשפט?"
docs = db.similarity_search(question, k=3)

print(f"\n--- תוצאות עבור: {question} ---")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")