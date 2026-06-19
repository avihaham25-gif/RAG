from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db", embedding_function=embeddings)

# שליפת דגימה מתוך ה-DB
results = db.get(include=['documents'])
print("--- תוכן ה-DB כפי שהוא שמור כרגע ---")
for doc in results['documents'][:3]: # מדפיס את 3 הקטעים הראשונים
    print(doc)
    print("-" * 20)