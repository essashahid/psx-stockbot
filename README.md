<div align="center">

## 📈 PSX StockBot

An AI-native, interactive stock analytics copilot for the Pakistan Stock Exchange.

`RAG` + `Charts` + `Analytics` — Upload your CSV, ask natural questions, get grounded answers and beautiful interactive visuals.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/) 
[![Plotly](https://img.shields.io/badge/Plotly-figure--factory-3C4CFF.svg)](https://plotly.com/python/) 
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-009688.svg)](https://faiss.ai/) 
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-sentence--transformers-6E57E0.svg)](https://www.sbert.net/) 
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-2E3A59.svg)](https://python.langchain.com/) 
[![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-4285F4.svg)](https://ai.google.dev/)

<!-- Replace with a real GIF/screenshot later -->
<img src="assets/hero.gif" alt="PSX StockBot demo" width="900" />

<br/>

### Make PSX data come alive — chat, chart, and crunch insights in seconds.

</div>

## ✨ Highlights

- **RAG chat over your CSV**: Ask precise questions; get answers grounded in your own data.
- **Interactive charts**: Line plots with moving averages (20/50/200) and candlestick views.
- **Zero-setup UX**: Use the repo’s default CSV or upload your own with one click.
- **Fast and memory-aware**: Smart sampling keeps huge CSVs responsive during embedding.
- **Analytics built-in**: Top movers by date, symbol browser, and more.

## 🧪 What can it do?

- "Which stocks had the highest volume on 2019-12-30?"
- "Plot KEL 2020-01 to 2020-03 as candlestick"
- "Chart LUCK and HUBC for 2021 with moving averages"
- "Show top movers for 2018-06-15"

## 🚀 Quickstart

1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure LLM

```bash
cp .env.example .env
# Edit .env and set
# GEMINI_API_KEY=your_key_here
```

3) Run

```bash
streamlit run app.py
```

4) Load data

- Sidebar → choose "Default CSV" (uses `psx_master_complete.csv` in the repo) or "Upload" your own.
- Adjust "Rows to index" and "Retriever depth (k)" to balance speed vs recall.

## 🧱 Data schema

Required columns in CSV:

`Date, Symbol, Company, Open, High, Low, Close, Volume`

- `Date`: any parseable format (YYYY-MM-DD recommended)
- `Symbol`: case-insensitive, normalized to uppercase
- Prices are downcast for memory efficiency

## 🧠 How it works

```mermaid
flowchart LR
  A[CSV File] --> B[Load + Optimize dtypes]
  B --> C[Textify rows]
  C --> D[Embeddings (Sentence Transformers)]
  D --> E[FAISS Vector Store]
  E --> F[Retriever (k)]
  F --> G[LLM (Gemini 1.5 Flash)]
  G --> H[Grounded Answer]

  B --> I[Charts Pipeline]
  I --> J[Parse Query -> Filter -> Plotly]
  J --> K[Interactive Charts]
```

Under the hood:

- Embeddings: `sentence-transformers/paraphrase-albert-small-v2`
- Vector store: FAISS
- LLM: Gemini 1.5 Flash (`google-generativeai`)
- App: Streamlit + Plotly

## 🖥️ UI at a glance

- **Ask**: RAG + Gemini answers with example prompt buttons.
- **Chart**: Natural-language charting with line/candlestick and built-in MAs.
- **Analytics**: Top movers by date and a symbol browser for quick exploration.

> Tip: For very large CSVs, start with 50k rows indexed and increase if needed.

## ⚙️ Configuration

Environment variables:

- `GEMINI_API_KEY`: required for LLM answers.

Performance toggles (Sidebar):

- "Rows to index for semantic search" (sampling)
- "Retriever depth (k)"

## 🔒 Privacy

- Your CSV is processed locally.
- Only small, relevant text snippets (retrieved context) plus your query are sent to the LLM.

## 🗺️ Roadmap

- Symbol synonyms and fuzzy matching
- Advanced analytics (RSI, MACD), strategy overlays
- Multi-file ingestion and persistent vector cache
- Export chart images and shareable reports

## 🤝 Contributing

PRs and issues are welcome! If you have a cool idea or find a bug, open an issue first to discuss.

## ❤️ Acknowledgements

- FAISS, Sentence-Transformers, LangChain, Plotly, Streamlit
- Gemini models by Google

---

If this project helps you explore PSX data faster, consider giving it a ⭐️!
