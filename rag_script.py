from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import time

# הגדרת המודל
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

all_docs = []

# רשימת נתיבים לסריקה: גם התיקייה הראשית וגם תיקיית המסמכים
folders_to_scan = [".", "./my_documents"]

for folder in folders_to_scan:
    for root, dirs, files in os.walk(folder):
        # מונע כניסה לתיקיית ה-DB כדי לא לקרוא קבצים מיותרים
        if "db" in dirs:
            dirs.remove("db")
            
        for file in files:
            if file.endswith(".docx") and not file.startswith("~$"): 
                file_path = os.path.join(root, file)
                try:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file_path
                    all_docs.extend(docs)
                    print(f"✅ נטען בהצלחה: {file}")
                except Exception as e:
                    print(f"❌ שגיאה בטעינת {file}: {e}")

# פיצול מותאם אישית: chunk_size=2000 ו-overlap=500 מבטיחים רצף לוגי
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=500, 
    separators=["\n\n", "\n", ".", "!", "?", " ", ""] 
)
texts = text_splitter.split_documents(all_docs)

# מחיקת ה-DB הישן ויצירת חדש
if os.path.exists("./db"):
    shutil.rmtree("./db")

print(f"מייצר אינדקס ל-{len(texts)} מקטעי טקסט (עם חפיפה רחבה)...")
db = Chroma.from_documents(texts, embeddings, persist_directory="./db")

# יצירת קובץ בקרה לסנכרון מול ממשק המשתמש
if not os.path.exists("./db"):
    os.makedirs("./db")
with open("./db/last_updated.txt", "w") as f:
    f.write(str(time.time()))

print("🎉 האינדקס נוצר בהצלחה, כולל כל הקבצים בתיקייה הראשית!")