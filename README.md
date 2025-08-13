## 📈 PSX StockBot

An interactive RAG-powered assistant for the Pakistan Stock Exchange. Upload your historical CSV, ask natural questions, and get:

- Conversational answers grounded in your data (Gemini)
- Interactive charts (Plotly): line, moving averages, candlestick
- Quick analytics like top movers per date

### ✨ Features
- Chat with your dataset using retrieval-augmented generation (HuggingFace embeddings + FAISS).
- Autodetect chart intents from queries like: "plot KEL 2020-01 to 2020-03 as candlestick".
- Clean, optimized CSV loading for multi-million row files with dtype downcasting.
- Streamlit UI with tabs for Ask, Chart, and Analytics.

### 🧱 Data schema
Expected columns in your CSV:
`Date, Symbol, Company, Open, High, Low, Close, Volume`

Dates can be any parseable format (YYYY-MM-DD recommended). Symbols are normalized to uppercase.

### 🚀 Quickstart

1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure your Gemini API key

```bash
cp .env.example .env
echo "GEMINI_API_KEY=YOUR_KEY" >> .env
```

3) Run

```bash
streamlit run app.py
```

Upload your CSV and start asking questions.

### 💡 Example prompts
- "Which stocks had the highest volume on 2019-12-30?"
- "Plot LUCK 2021 as candlestick"
- "Chart KEL 2020-01 to 2020-03"

### 🧠 Architecture
- Embeddings: `sentence-transformers/paraphrase-albert-small-v2`
- Vector store: FAISS
- LLM: Gemini 1.5 Flash (via `google-generativeai`)
- App: Streamlit + Plotly

### 🧪 Development
- Core logic lives in `main.py` (framework-agnostic).
- UI lives in `app.py` (Streamlit tabs: Ask, Chart, Analytics).
- Add tests or scripts as needed; `main.py` functions are pure and testable.

### 🔒 Privacy
Your data is processed locally. Only the RAG context snippets and your query are sent to Gemini when you click Answer.

### 🗺️ Roadmap ideas
- Symbol synonyms and fuzzy matching.
- Advanced analytics (RSI, MACD) and backtesting snippets.
- Multi-file ingestion and persistent vectorstore caching.
- Export chart images and report generation.

---

Made with ❤️ to explore PSX data in a powerful and intuitive way.
