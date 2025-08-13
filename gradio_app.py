import os
import tempfile
import pandas as pd
import gradio as gr
from main import (
    load_psx_csv,
    ensure_duckdb_loaded,
    load_psx_via_duckdb,
    build_retriever,
    ask_or_compute,
    generate_chart_from_query,
    top_movers,
)

# Lazy globals
_DF = None
_RETR = None
_DB = None


def init_data(mode: str):
    global _DF, _RETR, _DB
    default_path = os.path.join(os.path.dirname(__file__), "psx_master_complete.csv")
    if mode == "DuckDB (fast)" and os.path.exists(default_path):
        _DB = ensure_duckdb_loaded(default_path)
        _DF = load_psx_via_duckdb(_DB)
    elif os.path.exists(default_path):
        _DF = load_psx_csv(default_path)
    else:
        _DF = None
    if _DF is not None:
        _RETR = build_retriever(_DF)
    return f"Loaded {_DF.shape[0]:,} rows" if _DF is not None else "No data loaded"


def on_chat(message, history):
    # history is a list of dicts when type="messages"
    if history is None:
        history = []
    if _DF is None or _RETR is None:
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": "Please load data first (left panel)."}]
        return history
    answer = ask_or_compute(message, _RETR, _DF)
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return history


def on_chart(query):
    if _DF is None:
        return None, "Please load data first"
    fig, caption = generate_chart_from_query(query, _DF)
    if fig is None:
        return None, caption
    # Return Plotly figure directly for Gradio Plot component (no kaleido required)
    return fig, caption


def on_top_movers(date_str):
    if _DF is None:
        return "Please load data first"
    try:
        date = pd.to_datetime(date_str)
        res = top_movers(_DF, date, top_n=10)
        if res.empty:
            return "No data for selected date"
        return res[["Symbol", "Company", "Close", "PrevClose", "PctChange"]].round({"Close": 2, "PrevClose": 2, "PctChange": 2}).to_markdown(index=False)
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 PSX StockBot — Gradio UI")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["Default CSV", "DuckDB (fast)"], value="DuckDB (fast)", label="Data source")
            load_btn = gr.Button("Load data", variant="primary")
            load_out = gr.Markdown()
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", height=300)
            msg = gr.Textbox(label="Ask a question")
            ask = gr.Button("Ask", variant="primary")
            ask.click(on_chat, inputs=[msg, chatbot], outputs=chatbot)
            msg.submit(on_chat, inputs=[msg, chatbot], outputs=chatbot)
            gr.Markdown("## Chart")
            chart_q = gr.Textbox(label="Chart query", placeholder="plot KEL 2020-01 to 2020-03 as candlestick")
            render = gr.Button("Render Chart")
            img = gr.Plot()
            cap = gr.Markdown()
            render.click(on_chart, inputs=chart_q, outputs=[img, cap])
            gr.Markdown("## Top movers")
            date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value="2019-12-30")
            show = gr.Button("Show movers")
            movers = gr.Markdown()
            show.click(on_top_movers, inputs=date_in, outputs=movers)

    load_btn.click(lambda m: init_data(m), inputs=mode, outputs=load_out)

if __name__ == "__main__":
    demo.launch()


