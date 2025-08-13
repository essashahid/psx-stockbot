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
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import duckdb

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


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


# ---------- DuckDB accelerated path ----------


def ensure_duckdb_loaded(csv_path: str, db_path: str = "cache/psx.duckdb") -> str:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS psx AS SELECT * FROM read_csv_auto(?, HEADER=TRUE);
        """,
        [csv_path],
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_psx_symbol_date ON psx(Symbol, Date);")
    con.close()
    return db_path


def load_psx_via_duckdb(
    db_path: str,
    symbols: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    con = duckdb.connect(db_path)
    wheres = []
    params: List = []
    if symbols:
        placeholders = ",".join(["?"] * len(symbols))
        wheres.append(f"Symbol IN ({placeholders})")
        params.extend([s.upper() for s in symbols])
    if start:
        wheres.append("Date >= ?")
        params.append(start)
    if end:
        wheres.append("Date <= ?")
        params.append(end)
    where_sql = (" WHERE " + " AND ".join(wheres)) if wheres else ""
    sql = f"SELECT Date, Symbol, Company, Open, High, Low, Close, Volume FROM psx{where_sql} ORDER BY Symbol, Date"
    df = con.execute(sql, params).df()
    con.close()
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


def _compute_retriever_cache_key(df: pd.DataFrame, model_name: str, max_rows_for_index: int) -> str:
    try:
        min_date = str(pd.to_datetime(df["Date"]).min().date()) if "Date" in df.columns else ""
        max_date = str(pd.to_datetime(df["Date"]).max().date()) if "Date" in df.columns else ""
    except Exception:
        min_date = max_date = ""
    head_sig = df[["Symbol", "Date"]].astype(str).head(1000).to_csv(index=False) if set(["Symbol","Date"]).issubset(df.columns) else str(df.head(100))
    digest = hashlib.sha1(head_sig.encode()).hexdigest()[:12]
    key = f"{model_name.split('/')[-1]}_{len(df)}_{df['Symbol'].nunique() if 'Symbol' in df.columns else 0}_{min_date}_{max_date}_{max_rows_for_index}_{digest}"
    return key


def _save_vectorstore(vectorstore: FAISS, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(cache_dir))


def _load_vectorstore_if_exists(embedder: HuggingFaceEmbeddings, cache_dir: Path) -> Optional[FAISS]:
    if not cache_dir.exists():
        return None
    try:
        return FAISS.load_local(str(cache_dir), embedder, allow_dangerous_deserialization=True)
    except Exception:
        return None


def build_retriever(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_rows_for_index: int = 20000,
    k: int = 5,
):
    if df.empty:
        raise ValueError("DataFrame is empty; cannot build retriever")

    # Sample to keep memory reasonable on huge CSVs
    if max_rows_for_index and len(df) > max_rows_for_index:
        df = df.sample(n=max_rows_for_index, random_state=42)

    # Prefer GPU (Apple MPS) when available for faster first-run embedding
    device = "cpu"
    try:
        if torch is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = "mps"
    except Exception:
        device = "cpu"

    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64},
    )
    cache_key = _compute_retriever_cache_key(df, model_name, max_rows_for_index)
    cache_dir = Path("cache/indexes") / cache_key
    vs = _load_vectorstore_if_exists(embedder, cache_dir)
    if vs is None:
        docs: List[str] = [_row_to_doc(row) for _, row in df.iterrows()]
        vs = FAISS.from_texts(texts=docs, embedding=embedder)
        _save_vectorstore(vs, cache_dir)
    vectorstore = vs
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_retriever_with_progress(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_rows_for_index: int = 20000,
    k: int = 5,
    progress_cb: Optional[callable] = None,
):
    if df.empty:
        raise ValueError("DataFrame is empty; cannot build retriever")

    if max_rows_for_index and len(df) > max_rows_for_index:
        df = df.sample(n=max_rows_for_index, random_state=42)

    # Reuse embedder config (GPU/MPS if available)
    device = "cpu"
    try:
        if torch is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = "mps"
    except Exception:
        device = "cpu"
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64},
    )

    cache_key = _compute_retriever_cache_key(df, model_name, max_rows_for_index)
    cache_dir = Path("cache/indexes") / cache_key
    vs = _load_vectorstore_if_exists(embedder, cache_dir)
    if vs is not None:
        if progress_cb:
            progress_cb(1.0, "Loaded cached index")
        return vs.as_retriever(search_kwargs={"k": k})

    # Build incrementally to allow UI progress updates
    docs: List[str] = [_row_to_doc(row) for _, row in df.iterrows()]
    total = len(docs)
    batch_size = 1000
    vs = None
    for start in range(0, total, batch_size):
        batch = docs[start : start + batch_size]
        if start == 0:
            vs = FAISS.from_texts(texts=batch, embedding=embedder)
        else:
            vs.add_texts(batch)
        if progress_cb:
            ratio = min(0.99, (start + len(batch)) / total)
            progress_cb(ratio, f"Embedding batch {(start//batch_size)+1}/{(total+batch_size-1)//batch_size}")

    assert vs is not None
    _save_vectorstore(vs, cache_dir)
    if progress_cb:
        progress_cb(1.0, "Index built and cached")
    return vs.as_retriever(search_kwargs={"k": k})


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
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 256}
    )
    return response.text or "No answer generated."


def ask_bot_with_facts(query: str, retriever, facts_text: str) -> str:
    api_key = _ensure_gemini_configured()
    if not api_key:
        return (
            "Gemini API key not configured. Set GEMINI_API_KEY in your environment to enable answers."
        )

    docs = []
    if retriever:
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = retriever.get_relevant_documents(query)
    context = "\n".join([getattr(doc, "page_content", str(doc)) for doc in docs])

    system = (
        "You are PSX StockBot, a precise and concise financial assistant. "
        "Use the provided FACTS as the authoritative data. You may use CONTEXT for color, "
        "but do not contradict the FACTS. Answer in natural language for a general audience."
    )
    prompt = (
        f"{system}\n\nFACTS:\n{facts_text}\n\nCONTEXT:\n{context}\n\nQuestion:\n{query}\n\n"
        "Write a 2-4 sentence summary in clear, confident tone."
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 200}
    )
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


# ---------- Symbol performance summary rule ----------


def _format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}%"


def _format_num(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.2f}"


def _summarize_symbol(df: pd.DataFrame, symbol: str) -> Optional[str]:
    if df is None or df.empty:
        return None
    sdf = df[df["Symbol"].astype(str).str.upper() == symbol.upper()].sort_values("Date")
    if sdf.empty:
        return None
    start_row = sdf.iloc[0]
    end_row = sdf.iloc[-1]
    start_price = float(start_row["Close"])
    end_price = float(end_row["Close"])
    start_date = pd.to_datetime(start_row["Date"]).date()
    end_date = pd.to_datetime(end_row["Date"]).date()
    days = (pd.to_datetime(end_row["Date"]) - pd.to_datetime(start_row["Date"])) .days
    total_return_pct = (end_price - start_price) / start_price * 100 if start_price else None
    cagr_pct = None
    if days and days > 0 and start_price > 0:
        years = days / 365.0
        cagr_pct = (pow(end_price / start_price, 1.0 / years) - 1.0) * 100 if end_price > 0 else None
    high_idx = sdf["Close"].idxmax()
    low_idx = sdf["Close"].idxmin()
    high = float(sdf.loc[high_idx, "Close"]) if pd.notna(high_idx) else None
    low = float(sdf.loc[low_idx, "Close"]) if pd.notna(low_idx) else None
    high_date = pd.to_datetime(sdf.loc[high_idx, "Date"]).date() if pd.notna(high_idx) else None
    low_date = pd.to_datetime(sdf.loc[low_idx, "Date"]).date() if pd.notna(low_idx) else None
    avg_vol = float(sdf["Volume"].mean()) if "Volume" in sdf.columns else None

    lines: List[str] = []
    lines.append(f"Performance summary for {symbol} ({start_date} → {end_date})")
    rows = [
        {"Metric": "Start Close", "Value": _format_num(start_price)},
        {"Metric": "End Close", "Value": _format_num(end_price)},
        {"Metric": "Total Return", "Value": _format_pct(total_return_pct)},
        {"Metric": "CAGR (approx)", "Value": _format_pct(cagr_pct)},
        {"Metric": "High (date)", "Value": f"{_format_num(high)} ({high_date})" if high is not None else "—"},
        {"Metric": "Low (date)", "Value": f"{_format_num(low)} ({low_date})" if low is not None else "—"},
        {"Metric": "Avg Volume", "Value": f"{int(avg_vol):,}" if avg_vol is not None else "—"},
        {"Metric": "Trading days", "Value": f"{len(sdf):,}"},
    ]
    lines.append(_format_markdown_table(rows, ["Metric", "Value"]))
    return "\n".join(lines)


def rule_based_symbol_performance(query: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    q = query.lower()
    if not any(k in q for k in ["perform", "performance", "so far", "how has", "trend"]):
        return ""
    known_symbols = sorted(df["Symbol"].dropna().astype(str).str.upper().unique().tolist())
    syms = _maybe_extract_symbols(query, known_symbols)
    if not syms:
        return ""
    sym = syms[0]
    start, end = _maybe_extract_dates(query)
    narrative = summarize_symbol_narrative(df, sym, start, end)
    if narrative:
        return narrative
    table = _summarize_symbol(df, sym)
    return table or ""


def summarize_symbol_narrative(
    df: pd.DataFrame,
    symbol: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    sdf = df[df["Symbol"].astype(str).str.upper() == symbol.upper()].sort_values("Date")
    if start is not None:
        sdf = sdf[sdf["Date"] >= start]
    if end is not None:
        sdf = sdf[sdf["Date"] <= end]
    if sdf.empty:
        return None
    start = sdf.iloc[0]
    end = sdf.iloc[-1]
    start_price = float(start["Close"])
    end_price = float(end["Close"])
    start_date = pd.to_datetime(start["Date"]).date()
    end_date = pd.to_datetime(end["Date"]).date()
    total_return_pct = (end_price - start_price) / start_price * 100 if start_price else None
    high = float(sdf["Close"].max())
    low = float(sdf["Close"].min())

    dir_word = "increased" if total_return_pct is not None and total_return_pct >= 0 else "declined"
    magnitude = f"about {abs(total_return_pct):.1f}%" if total_return_pct is not None else "over the observed period"

    return (
        f"From {start_date} to {end_date}, {symbol} {dir_word} by {magnitude}, "
        f"moving from {start_price:.2f} to {end_price:.2f}. Over this span, it traded as high as {high:.2f} "
        f"and as low as {low:.2f}."
    )


def compose_symbol_facts(
    df: pd.DataFrame,
    symbol: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    sdf = df[df["Symbol"].astype(str).str.upper() == symbol.upper()].sort_values("Date")
    if start is not None:
        sdf = sdf[sdf["Date"] >= start]
    if end is not None:
        sdf = sdf[sdf["Date"] <= end]
    if sdf.empty:
        return None
    start = sdf.iloc[0]
    end = sdf.iloc[-1]
    start_price = float(start["Close"])
    end_price = float(end["Close"])
    start_date = pd.to_datetime(start["Date"]).date()
    end_date = pd.to_datetime(end["Date"]).date()
    total_return_pct = (end_price - start_price) / start_price * 100 if start_price else None
    high = float(sdf["Close"].max())
    low = float(sdf["Close"].min())
    avg_vol = float(sdf["Volume"].mean()) if "Volume" in sdf.columns else None
    return (
        f"Symbol: {symbol}\n"
        f"DateRange: {start_date} -> {end_date}\n"
        f"StartClose: {start_price:.2f}\n"
        f"EndClose: {end_price:.2f}\n"
        f"TotalReturnPct: {total_return_pct:.2f}\n"
        f"High: {high:.2f}\n"
        f"Low: {low:.2f}\n"
        f"AvgVolume: {int(avg_vol) if avg_vol is not None else 'NA'}\n"
        f"TradingDays: {len(sdf)}"
    )


def ask_or_compute(query: str, retriever, df: Optional[pd.DataFrame]) -> str:  # type: ignore[no-redef]
    # Prefer natural language from LLM; pass concise facts when we detect a symbol performance ask
    if df is not None and not df.empty:
        q = query.lower()
        if any(k in q for k in ["perform", "performance", "so far", "how has", "trend"]):
            known_symbols = sorted(df["Symbol"].dropna().astype(str).str.upper().unique().tolist())
            syms = _maybe_extract_symbols(query, known_symbols)
            if syms:
                _start, _end = _maybe_extract_dates(query)
                facts = compose_symbol_facts(df, syms[0], _start, _end)
                if facts:
                    return ask_bot_with_facts(query, retriever, facts)
    # Then try other deterministic rules
    deterministic = rule_based_answer(query, df if df is not None else pd.DataFrame())
    if deterministic:
        return deterministic
    # Fallback to LLM
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
    """Extract date range from text.

    - YYYY-MM-DD → single day
    - YYYY-MM → whole month
    - YYYY → whole year
    - Multiple matches collapse to [min, max]
    """
    # Full dates
    full = [pd.to_datetime(x) for x in re.findall(r"(20\d{2}-\d{2}-\d{2})", text)]
    if full:
        full.sort()
        return full[0], full[-1]
    # Year-month
    ym = re.findall(r"(20\d{2}-\d{2})", text)
    if ym:
        dt = pd.to_datetime(ym[0] + "-01")
        start = dt.normalize()
        end = (start + pd.offsets.MonthEnd(1)).normalize()
        return start, end
    # Year
    y = re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)
    if y:
        yr = int(y[0])
        return pd.Timestamp(year=yr, month=1, day=1), pd.Timestamp(year=yr, month=12, day=31)
    return None, None


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

