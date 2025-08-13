import streamlit as st
import pandas as pd
from io import BytesIO
from main import (
    load_psx_csv,
    build_retriever,
    ask_bot,
    generate_chart_from_query,
    top_movers,
)

st.set_page_config(page_title="📈 PSX StockBot", layout="wide")
st.title("📈 PSX StockBot: PSX Chat, Charts, and Retrieval")
st.caption("Upload PSX CSV and ask questions. Try: 'Chart KEL 2020-01 to 2020-03', 'Top movers 2019-12-30'")

with st.sidebar:
    uploaded_file = st.file_uploader("📄 Upload PSX data CSV", type=["csv"])
    st.markdown(
        "- Expected columns: `Date, Symbol, Company, Open, High, Low, Close, Volume`\n"
        "- LLM answers require `GEMINI_API_KEY` in your environment."
    )

@st.cache_data(show_spinner=False)
def _load_df_cached(file_bytes: bytes):
    return load_psx_csv(BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def _build_retriever_cached(df: pd.DataFrame):
    return build_retriever(df)


df = None
retriever = None
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        df = _load_df_cached(file_bytes)
        retriever = _build_retriever_cached(df)
        st.success(f"Loaded {len(df):,} rows · {df['Symbol'].nunique()} symbols")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

tab1, tab2, tab3 = st.tabs(["Ask", "Chart", "Analytics"])

with tab1:
    query = st.text_input(
        "Ask a question",
        placeholder="E.g. Which stocks had the highest volume on 2019-12-30?",
    )
    if st.button("Answer", type="primary"):
        if df is None or retriever is None:
            st.warning("Please upload a CSV first.")
        else:
            with st.spinner("Thinking..."):
                answer = ask_bot(query, retriever)
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
