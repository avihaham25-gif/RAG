from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3")
db = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

question = input("מה תרצה לשאול את הצ'אט? ")
docs = retriever.invoke(question)
context_text = "\n\n".join([doc.page_content for doc in docs])

prompt_template = ChatPromptTemplate.from_template("""
ענה על השאלה הבאה בהתבסס על ההקשר המצורף בלבד. ענה ללא הקדמות. מבנה חובה: 1. כותרת קצרה. 2. בשורה חדשה, פסקה אחת ויחידה המכילה את התשובה והציטוט הרלוונטי.
{context}

שאלה: {input}
""")

formatted_prompt = prompt_template.format(context=context_text, input=question)
response = llm.invoke(formatted_prompt)

print("\n--- תשובת הצ'אט ---")
print(response.content)