# app.py
import streamlit as st

# ⬇️  FIRST Streamlit command
st.set_page_config(page_title="📈 PSX Stock Chatbot", layout="centered")

# Only import main *after* page-config is set
from main import ask_bot  # noqa: E402

st.title("📈 PSX Stock Chatbot")
st.markdown("Ask me anything about the Pakistan Stock Exchange (2015–2025)")

query = st.text_input(
    "Your question", placeholder="E.g. What stocks had highest volume on Dec 30 2015?"
)

if query:
    with st.spinner("Thinking..."):
        answer = ask_bot(query)
    st.markdown("### 📊 Answer")
    st.write(answer)
