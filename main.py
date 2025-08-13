"""
Core logic for PSX StockBot: data loading, retrieval, LLM QA, and chart utilities.

This module is intentionally framework-agnostic so it can be used from
Streamlit, a CLI, or tests.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------- Environment / LLM configuration ----------

_ENV_LOADED = False


def _ensure_gemini_configured() -> Optional[str]:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    return api_key


# ---------- Data loading & optimization ----------


EXPECTED_COLUMNS = [
    "Date",
    "Symbol",
    "Company",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse dates
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Downcast numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("int64")

    # Normalize symbol/company text
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    if "Company" in df.columns:
        df["Company"] = df["Company"].astype(str).str.strip()

    # Drop rows with missing critical fields
    critical = [c for c in ["Date", "Symbol", "Close"] if c in df.columns]
    if critical:
        df = df.dropna(subset=critical)
    return df


def load_psx_csv(path_or_buffer) -> pd.DataFrame:
    if isinstance(path_or_buffer, pd.DataFrame):
        df = path_or_buffer
    else:
        df = pd.read_csv(path_or_buffer)
    return _optimize_dtypes(df)


# ---------- Retrieval (RAG) ----------


def _row_to_doc(row: pd.Series) -> str:
    parts: List[str] = []
    if pd.notna(row.get("Date")):
        parts.append(f"On {pd.to_datetime(row['Date']).date()}")
    sym = row.get("Symbol", "?")
    comp = row.get("Company", "")
    if comp:
        parts.append(f"{sym} ({comp})")
    else:
        parts.append(str(sym))
    o = row.get("Open", None)
    h = row.get("High", None)
    l = row.get("Low", None)
    c = row.get("Close", None)
    v = row.get("Volume", None)
    details = []
    if pd.notna(o):
        details.append(f"open {o}")
    if pd.notna(h):
        details.append(f"high {h}")
    if pd.notna(l):
        details.append(f"low {l}")
    if pd.notna(c):
        details.append(f"close {c}")
    if pd.notna(v):
        details.append(f"volume {int(v)}")
    if details:
        parts.append(", ".join(details))
    return ", ".join(parts) + "."


def build_retriever(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-albert-small-v2",
    max_rows_for_index: int = 50000,
    k: int = 8,
):
    if df.empty:
        raise ValueError("DataFrame is empty; cannot build retriever")

    # Sample to keep memory reasonable on huge CSVs
    if max_rows_for_index and len(df) > max_rows_for_index:
        df = df.sample(n=max_rows_for_index, random_state=42)

    docs: List[str] = [_row_to_doc(row) for _, row in df.iterrows()]

    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=docs, embedding=embedder)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ---------- LLM QA ----------


def ask_bot(query: str, retriever) -> str:
    api_key = _ensure_gemini_configured()
    if not api_key:
        return (
            "Gemini API key not configured. Set GEMINI_API_KEY in your environment to enable answers."
        )

    # LangChain retrievers are Runnable in newer versions; prefer invoke
    docs = []
    if retriever:
        try:
            docs = retriever.invoke(query)
        except Exception:
            # Back-compat
            docs = retriever.get_relevant_documents(query)
    context = "\n".join([getattr(doc, "page_content", str(doc)) for doc in docs])

    system = (
        "You are PSX StockBot, a precise and concise financial data assistant. "
        "When possible, cite dates and symbols explicitly. If the user asks for a chart, "
        "respond with a short textual explanation; chart rendering is handled by the UI."
    )

    prompt = (
        f"{system}\n\nContext:\n{context}\n\nUser question:\n{query}\n\n"
        "Answer clearly in 1-2 short paragraphs or a bulleted list."
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text or "No answer generated."


def _format_markdown_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n" + "\n".join(lines)


def rule_based_answer(query: str, df: pd.DataFrame, top_n: int = 10) -> str:
    """Deterministic answers for common analytics questions.

    Currently supports:
      - Highest/Top volume on a specific date: e.g., "highest volume on 2019-12-30".
    """
    if df is None or df.empty:
        return ""

    q = query.lower()
    date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", q)
    asks_volume = any(k in q for k in ["highest volume", "top volume", "most volume", "max volume"])
    if date_match and asks_volume:
        when = pd.to_datetime(date_match.group(1))
        day = df[df["Date"] == when]
        if day.empty:
            return f"No rows found for {when.date()} in the dataset."
        day = day.sort_values("Volume", ascending=False).head(top_n)
        rows = [
            {
                "Symbol": s,
                "Company": c,
                "Volume": f"{int(v):,}",
            }
            for s, c, v in zip(day["Symbol"], day.get("Company", ""), day["Volume"])
        ]
        table = _format_markdown_table(rows, ["Symbol", "Company", "Volume"])
        return f"Top {len(day)} by Volume on {when.date()}:" + table
    return ""


def ask_or_compute(query: str, retriever, df: Optional[pd.DataFrame]) -> str:
    deterministic = rule_based_answer(query, df if df is not None else pd.DataFrame())
    if deterministic:
        return deterministic
    return ask_bot(query, retriever)


# ---------- Query parsing and charts ----------


_CHART_KEYWORDS = ["chart", "plot", "line", "candlestick", "ohlc"]


def _maybe_extract_symbols(text: str, known_symbols: Optional[List[str]]) -> List[str]:
    if not known_symbols:
        return []
    text_u = text.upper()
    hits = [s for s in known_symbols if s in text_u]
    return list(dict.fromkeys(hits))  # unique preserving order


def _maybe_extract_dates(text: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    # Very lightweight date extractor: YYYY, YYYY-MM, or YYYY-MM-DD
    date_patterns = [
        r"(20\d{2}-\d{2}-\d{2})",
        r"(20\d{2}-\d{2})",
        r"(20\d{2})",
    ]
    found: List[pd.Timestamp] = []
    for pat in date_patterns:
        for m in re.findall(pat, text):
            try:
                found.append(pd.to_datetime(m))
            except Exception:
                pass
    if not found:
        return None, None
    found.sort()
    if len(found) == 1:
        return found[0], None
    return found[0], found[-1]


def parse_query_for_chart(query: str, known_symbols: Optional[List[str]]) -> Dict:
    q = query.lower()
    is_chart = any(k in q for k in _CHART_KEYWORDS)
    symbols = _maybe_extract_symbols(query, known_symbols)
    start, end = _maybe_extract_dates(query)
    chart_type = "candlestick" if "candlestick" in q or "ohlc" in q else "line"
    return {
        "is_chart": is_chart,
        "symbols": symbols,
        "start": start,
        "end": end,
        "chart_type": chart_type,
    }


def _filter_df(
    df: pd.DataFrame,
    symbols: Optional[List[str]] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    f = df
    if symbols:
        f = f[f["Symbol"].isin(symbols)]
    if start is not None:
        f = f[f["Date"] >= start]
    if end is not None:
        f = f[f["Date"] <= end]
    return f.sort_values(["Symbol", "Date"])  # stable plotting order


def make_price_line_chart(df: pd.DataFrame, title: str = "Close Price") -> go.Figure:
    fig = px.line(df, x="Date", y="Close", color="Symbol", title=title)
    fig.update_layout(legend_title_text="Symbol", hovermode="x unified")
    return fig


def make_candlestick_chart(df: pd.DataFrame, title: str = "OHLC") -> go.Figure:
    # Expect one symbol for candlestick; if multiple, pick the first
    symbols = sorted(df["Symbol"].unique().tolist())
    sym = symbols[0]
    sdf = df[df["Symbol"] == sym]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=sdf["Date"],
                open=sdf["Open"],
                high=sdf["High"],
                low=sdf["Low"],
                close=sdf["Close"],
                name=sym,
            )
        ]
    )
    fig.update_layout(title=f"{title} — {sym}", xaxis_title="Date", yaxis_title="Price")
    return fig


def add_moving_averages(fig: go.Figure, df: pd.DataFrame, windows: List[int]) -> go.Figure:
    symbols = df["Symbol"].unique().tolist()
    for sym in symbols:
        sdf = df[df["Symbol"] == sym].sort_values("Date")
        for w in windows:
            ma = sdf["Close"].rolling(window=w, min_periods=max(1, w // 2)).mean()
            fig.add_trace(
                go.Scatter(
                    x=sdf["Date"],
                    y=ma,
                    mode="lines",
                    name=f"{sym} MA{w}",
                    line=dict(width=1, dash="dash"),
                    hovertemplate="%{y:.2f}",
                )
            )
    return fig


def generate_chart_from_query(query: str, df: pd.DataFrame) -> Tuple[Optional[go.Figure], str]:
    symbols = sorted(df["Symbol"].dropna().astype(str).str.upper().unique().tolist())
    parsed = parse_query_for_chart(query, symbols)
    if not parsed["is_chart"]:
        return None, ""

    f = _filter_df(df, parsed["symbols"], parsed["start"], parsed["end"])
    if f.empty:
        return None, "No data matches your query (check symbols or date range)."

    if parsed["chart_type"] == "candlestick":
        fig = make_candlestick_chart(f)
    else:
        fig = make_price_line_chart(f)
        fig = add_moving_averages(fig, f, windows=[20, 50, 200])

    caption = (
        f"Chart type: {parsed['chart_type']}. Symbols: {', '.join(parsed['symbols'] or ['(auto)'])}. "
        f"Date range: {parsed['start'].date() if parsed['start'] else 'min'} → {parsed['end'].date() if parsed['end'] else 'max'}."
    )
    return fig, caption


# ---------- Analytics helpers ----------


def top_movers(df: pd.DataFrame, date: pd.Timestamp, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    # Compute previous close per symbol (previous trading day, not calendar day)
    s = df.sort_values(["Symbol", "Date"]).copy()
    s["PrevClose"] = s.groupby("Symbol")["Close"].shift(1)
    day = s[s["Date"] == pd.to_datetime(date)].copy()
    if day.empty:
        return day
    day["PctChange"] = (day["Close"] - day["PrevClose"]) / day["PrevClose"] * 100
    day["PctChange"] = day["PctChange"].replace([np.inf, -np.inf], np.nan)
    day = day.sort_values(["PctChange", "Volume"], ascending=[False, False])
    return day.head(top_n)

