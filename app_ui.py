import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.markdown("""
    <style>
        [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
        [data-testid="stChatInput"]   { direction: rtl; text-align: right; }
        div[data-testid="stChatMessage"] p { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)

def check_sync():
    return os.path.exists("./db/last_updated.txt")

st.title("מנוע שליפת נתונים ממסמכים")

if not check_sync():
    st.error("⚠️ האינדקס לא מעודכן! אנא הרץ את rag_script.py בטרמינל.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="./db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 6})

@st.cache_resource
def get_llm():
    return ChatOllama(
        model="mistral",
        temperature=0,
        timeout=60,
        num_predict=512,
    )

def check_ollama():
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

if not check_ollama():
    st.error("❌ Ollama לא רץ! פתח טרמינל והרץ: `ollama serve`")
    st.stop()

retriever = get_retriever()
llm = get_llm()

# ✅ תיקון שפה: הוראת עברית מופיעה בסוף ה-human turn (recency bias של Mistral)
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "אתה עוזר המשיב אך ורק בעברית. "
        "חל איסור מוחלט לענות באנגלית או בכל שפה אחרת. "
        "השתמש אך ורק במידע שמופיע ב-[Context] שלהלן. "
        "אם התשובה אינה במסמכים, השב: 'המידע אינו קיים במסמכים'."
    )),
    ("human", (
        "[Context]\n{context}\n\n"
        "[שאלה]\n{question}\n\n"
        # ✅ ההוראה בסוף — Mistral נותן משקל לטקסט האחרון
        "⚠️ חובה: כתוב את כל תשובתך בעברית בלבד, גם אם ה-Context באנגלית. "
        "אסור בהחלט להשתמש באנגלית. תרגם מושגים מהמסמך לעברית בתוך תשובתך. "
        "ענה ללא הקדמות. מבנה חובה: 1. כותרת קצרה. 2. בשורה חדשה, פסקה אחת ויחידה המכילה את התשובה והציטוט הרלוונטי."
    ))
])

chain = prompt | llm | StrOutputParser()

# הצגת היסטוריה
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            f"<div style='direction: rtl; text-align: right;'>{message['content']}</div>",
            unsafe_allow_html=True
        )

if user_input := st.chat_input("הקלד שאלה..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(
            f"<div style='direction: rtl; text-align: right;'>{user_input}</div>",
            unsafe_allow_html=True
        )

    with st.chat_message("assistant"):
        try:
            with st.spinner("מחפש ומנתח..."):
                docs = retriever.invoke(user_input)

                with st.expander("ראה מקטעי מקור שנשלפו"):
                    st.write([d.page_content for d in docs])

                context_text = "\n\n".join([doc.page_content for doc in docs])

                # ✅ תיקון שפה: הערה למודל שה-Context עשוי להיות באנגלית
                context_with_note = (
                    "[המסמכים הבאים עשויים להיות בעברית או באנגלית — "
                    "התשובה שלך חייבת להיות בעברית בלבד]\n\n"
                    + context_text
                )

                response_placeholder = st.empty()
                full_response = ""

                for chunk in chain.stream({"context": context_with_note, "question": user_input}):
                    full_response += chunk
                    response_placeholder.markdown(
                        f"<div style='direction: rtl; text-align: right;'>{full_response}▌</div>",
                        unsafe_allow_html=True
                    )

                response_placeholder.markdown(
                    f"<div style='direction: rtl; text-align: right;'>{full_response}</div>",
                    unsafe_allow_html=True
                )

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            err_msg = f"❌ שגיאה: {str(e)}"
            st.error(err_msg)
            if "timeout" in str(e).lower() or "connect" in str(e).lower():
                st.info("💡 נסה: `ollama serve` בטרמינל, ובדוק ש-Mistral מותקן: `ollama pull mistral`")