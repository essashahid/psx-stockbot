# app.py
import streamlit as st
import pandas as pd
from main import build_retriever, ask_bot

st.set_page_config(page_title="📈 PSX Stock Chatbot", layout="centered")
st.title("📈 PSX Stock Chatbot")
st.markdown("Ask me anything about the Pakistan Stock Exchange (2015–2025)")

uploaded_file = st.file_uploader("📄 Upload your PSX data CSV", type=["csv"])

retriever = None
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    retriever = build_retriever(df)

query = st.text_input(
    "Your question", placeholder="E.g. What stocks had highest volume on Dec 30 2015?"
)

if query:
    if retriever is None:
        st.warning("Please upload your CSV before asking a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = ask_bot(query, retriever)
                st.markdown("### 📊 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
