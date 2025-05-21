# app.py
import streamlit as st
from main import ask_bot  # noqa: E402  (import after streamlit OK)

st.set_page_config(page_title="📈 PSX Stock Chatbot", layout="centered")

st.write("🛠  app.py loaded")  # Now safe — after set_page_config

st.title("📈 PSX Stock Chatbot")
st.markdown("Ask me anything about the Pakistan Stock Exchange (2015–2025)")

query = st.text_input(
    "Your question", placeholder="E.g. What stocks had highest volume on Dec 30 2015?"
)

if query:
    st.write("💬 Received query:", query)
    with st.spinner("Thinking..."):
        try:
            answer = ask_bot(query)
            st.markdown("### 📊 Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
