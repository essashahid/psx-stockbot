# main.py
import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("🔧 main.py loaded")

# Load Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 🔄 Upload CSV file via Streamlit
uploaded_file = st.file_uploader("📄 Upload your PSX data CSV", type=["csv"])
if not uploaded_file:
    st.warning("Please upload your `psx_master_complete.csv` to continue.")
    st.stop()

# Load the uploaded CSV
df = pd.read_csv(uploaded_file, parse_dates=["Date"])

# Prepare document text from rows
df["doc"] = df.apply(
    lambda r: f"On {r.Date.date()}, {r.Symbol} ({r.Company}) opened at {r.Open}, high {r.High}, low {r.Low}, closed at {r.Close}, volume was {r.Volume}.",
    axis=1,
)

# Embed the docs
model_name = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=model_name)
docs = df["doc"].tolist()
doc_embeddings = embedder.embed_documents(docs)

# Build FAISS index
dimension = len(doc_embeddings[0])
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(doc_embeddings).astype("float32"))

vectorstore = LangFAISS(embedding_function=embedder, index=faiss_index, documents=docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# Gemini-powered question answering
def ask_bot(query: str) -> str:
    print(f"🔍 ask_bot received query: {query}")
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = (
        f"You are a helpful stock market assistant.\n\n"
        f"Here are some historical records:\n{context}\n\n"
        f"Answer the user’s question:\n{query}"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    print("✔️  Gemini responded")
    return response.text
