import os
import streamlit as st
import pandas as pd
from io import BytesIO
from main import (
    load_psx_csv,
    ensure_duckdb_loaded,
    load_psx_via_duckdb,
    build_retriever,
    build_retriever_with_progress,
    ask_or_compute,
    generate_chart_from_query,
    top_movers,
)

st.set_page_config(page_title="📈 PSX StockBot", layout="wide")
st.title("📈 PSX StockBot: PSX Chat, Charts, and Retrieval")
st.caption("Upload PSX CSV and ask questions. Try: 'Chart KEL 2020-01 to 2020-03', 'Top movers 2019-12-30'")

with st.sidebar:
    data_mode = st.radio("Data source", ["Upload", "Default CSV", "DuckDB (fast)"], horizontal=True)
    uploaded_file = None
    default_path = os.path.join(os.path.dirname(__file__), "psx_master_complete.csv")
    if data_mode == "Upload":
        uploaded_file = st.file_uploader("📄 Upload PSX data CSV", type=["csv"])
    else:
        if os.path.exists(default_path):
            st.success("Using repository CSV: psx_master_complete.csv")
        else:
            st.warning("Default CSV not found in repo. Switch to Upload.")

    max_rows_for_index = st.slider(
        "Rows to index for semantic search",
        min_value=10000,
        max_value=200000,
        step=10000,
        value=50000,
        help="Sampling keeps memory usage stable on huge CSVs",
    )
    retriever_k = st.slider("Retriever depth (k)", 3, 20, 8)
    st.markdown(
        "- Expected columns: `Date, Symbol, Company, Open, High, Low, Close, Volume`\n"
        "- LLM answers require `GEMINI_API_KEY` in your environment."
    )

@st.cache_data(show_spinner=False)
def _load_df_cached(file_bytes: bytes):
    return load_psx_csv(BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def _build_retriever_cached(df: pd.DataFrame, max_rows_for_index: int, k: int):
    # Use a Streamlit progress bar if building from scratch
    progress = st.progress(0.0, text="Preparing index…")

    def cb(ratio: float, msg: str):
        progress.progress(ratio, text=msg)

    try:
        return build_retriever_with_progress(df, max_rows_for_index=max_rows_for_index, k=k, progress_cb=cb)
    finally:
        progress.empty()


df = None
retriever = None
try:
    if data_mode == "Upload" and uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        df = _load_df_cached(file_bytes)
    elif data_mode == "Default CSV" and os.path.exists(default_path):
        df = load_psx_csv(default_path)
    elif data_mode == "DuckDB (fast)" and os.path.exists(default_path):
        db_path = ensure_duckdb_loaded(default_path)
        df = load_psx_via_duckdb(db_path)
    if df is not None:
        retriever = _build_retriever_cached(df, max_rows_for_index, retriever_k)
        st.success(f"Loaded {len(df):,} rows · {df['Symbol'].nunique()} symbols")
except Exception as e:
    st.error(f"Failed to load data: {e}")

tab1, tab2, tab3 = st.tabs(["Ask", "Chart", "Analytics"])

with tab1:
    q_col, b_col = st.columns([4, 1])
    with q_col:
        query = st.text_input(
            "Ask a question",
            placeholder="E.g. Which stocks had the highest volume on 2019-12-30?",
            key="ask_query",
        )
    with b_col:
        ask_clicked = st.button("Ask", type="primary")

    cols = st.columns(3)
    if cols[0].button("Example: Highest volume 2019-12-30"):
        query = "Which stocks had the highest volume on 2019-12-30?"
        st.rerun()
    if cols[1].button("Example: Avg close for KEL 2020"):
        query = "What was the average close for KEL in 2020?"
        st.rerun()
    if cols[2].button("Example: Rising stocks 2018-06"):
        query = "Which symbols were rising in 2018-06?"
        st.rerun()
    if ask_clicked:
        if df is None or retriever is None:
            st.warning("Please upload a CSV first.")
        else:
            st.toast("Processing your question…", icon="🤖")
            with st.spinner("Retrieving data and composing answer…"):
                answer = ask_or_compute(query, retriever, df)
            st.markdown("### 📊 Answer")
            st.write(answer)

with tab2:
    chart_query = st.text_input(
        "Chart query",
        placeholder="e.g. plot KEL 2020-01 to 2020-03 as candlestick",
        key="chart_query",
    )
    if st.button("Render Chart"):
        if df is None:
            st.warning("Please upload a CSV first.")
        else:
            with st.spinner("Rendering chart..."):
                fig, caption = generate_chart_from_query(chart_query, df)
            if fig is None:
                st.info(caption or "No chart requested in query.")
            else:
                st.plotly_chart(fig, use_container_width=True)
                if caption:
                    st.caption(caption)

with tab3:
    if df is None:
        st.info("Upload CSV to enable analytics.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Movers")
            date = st.date_input("Pick a date", value=pd.to_datetime(df["Date"].min()).date())
            movers = top_movers(df, pd.to_datetime(date))
            if movers.empty:
                st.info("No data for selected date.")
            else:
                st.dataframe(
                    movers[["Symbol", "Company", "Close", "PrevClose", "PctChange"]]
                    .round({"Close": 2, "PrevClose": 2, "PctChange": 2})
                    .rename(columns={"PctChange": "% Change"}),
                    use_container_width=True,
                )
        with col2:
            st.subheader("Symbol Browser")
            sym = st.selectbox("Symbol", sorted(df["Symbol"].unique().tolist()))
            sdf = df[df["Symbol"] == sym].sort_values("Date")
            st.dataframe(sdf.tail(50), use_container_width=True)
