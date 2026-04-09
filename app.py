"""
Geo-Agentic RAG — Subsurface AI Assistant
Streamlit application for the Volve field production analysis.
"""
import os

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# --- Page Config (must be first Streamlit command) ---
st.set_page_config(
    page_title="Geo-Agentic RAG | Volve Field",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_streamlit_secrets_to_environ() -> None:
    """Streamlit Community Cloud stores deploy secrets in st.secrets; mirror into os.environ
    before importing src.config (which reads OPENAI_API_KEY, etc.)."""
    try:
        secrets = st.secrets
    except (FileNotFoundError, RuntimeError, TypeError):
        return
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        try:
            if key not in secrets:
                continue
            val = secrets[key]
            if val is None:
                continue
            s = str(val).strip()
            if s:
                os.environ[key] = s
        except Exception:
            continue


_apply_streamlit_secrets_to_environ()

from langchain_core.messages import AIMessage, HumanMessage

from src.data_loader import load_daily_production, load_monthly_production, get_well_list, get_well_summary
from src.anomaly import detect_anomalies
from src.agent import create_agent

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E3A5F;
    }
    .anomaly-critical { color: #dc3545; font-weight: bold; }
    .anomaly-high { color: #fd7e14; font-weight: bold; }
    .anomaly-medium { color: #ffc107; }
    .source-citation {
        background: #f0f4f8;
        border-left: 3px solid #4A90D9;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    /* Keep chat input visible at the bottom of the scrollable main area */
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 2;
        padding-top: 0.5rem;
        background-color: var(--background-color);
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading (cached) ---
@st.cache_data
def load_data():
    daily = load_daily_production()
    monthly = load_monthly_production()
    return daily, monthly


@st.cache_data
def get_anomalies(_daily_df, well_name=None):
    return detect_anomalies(_daily_df, well_name)


@st.cache_resource
def get_agent(_daily_df):
    return create_agent(_daily_df)


# --- Load Data ---
try:
    daily_df, monthly_df = load_data()
    wells = get_well_list(daily_df)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure 'Volve production data.xlsx' is in the data/production/ folder.")
    data_loaded = False
    wells = []


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛢️ Geo-Agentic RAG")
    st.markdown("*Subsurface AI Assistant*")
    st.markdown("---")

    if data_loaded:
        selected_well = st.selectbox(
            "Select Well",
            options=["All Wells"] + wells,
            index=0,
        )

        st.markdown("---")
        st.markdown("#### Date Range")
        min_date = daily_df["DATEPRD"].min().date()
        max_date = daily_df["DATEPRD"].max().date()

        date_range = st.date_input(
            "Filter dates",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("---")
        st.markdown("#### Quick Stats")
        summary = get_well_summary(daily_df)
        st.metric("Total Wells", len(wells))
        st.metric("Production Days", f"{len(daily_df):,}")
        st.metric("Total Oil (Sm3)", f"{daily_df['BORE_OIL_VOL'].sum():,.0f}")

        st.markdown("---")
        st.markdown("#### About")
        st.markdown(
            "Built as a portfolio project demonstrating "
            "agentic RAG for subsurface workflows. "
            "Uses real data from Equinor's Volve field."
        )

if not data_loaded:
    st.stop()

# --- Filter Data ---
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = daily_df[
        (daily_df["DATEPRD"].dt.date >= start_date) &
        (daily_df["DATEPRD"].dt.date <= end_date)
    ]
else:
    filtered_df = daily_df

if selected_well != "All Wells":
    plot_df = filtered_df[filtered_df["WELL_NAME"] == selected_well]
else:
    plot_df = filtered_df

# --- Main Content ---
st.markdown('<p class="main-header">🛢️ Volve Field — Subsurface AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Agentic RAG combining production data with well documentation</p>', unsafe_allow_html=True)

# --- Tabs ---
tab_dashboard, tab_chat, tab_anomalies = st.tabs([
    "📊 Production Dashboard",
    "💬 AI Assistant",
    "⚠️ Anomaly Detection"
])


# --- TAB 1: Dashboard ---
with tab_dashboard:
    if selected_well == "All Wells":
        # Multi-well overview
        st.markdown("### Field Production Overview")

        # Oil production by well
        fig = px.area(
            plot_df.groupby(["DATEPRD", "WELL_NAME"])["BORE_OIL_VOL"].sum().reset_index(),
            x="DATEPRD", y="BORE_OIL_VOL", color="WELL_NAME",
            title="Daily Oil Production by Well (Sm3)",
            labels={"DATEPRD": "Date", "BORE_OIL_VOL": "Oil Volume (Sm3)", "WELL_NAME": "Well"},
        )
        fig.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Well summary table
        st.markdown("### Well Summary")
        summary = get_well_summary(filtered_df)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    else:
        # Single well detail view
        st.markdown(f"### Well: {selected_well}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Oil (Sm3)", f"{plot_df['BORE_OIL_VOL'].sum():,.0f}")
        with col2:
            st.metric("Avg Water Cut", f"{plot_df['WATER_CUT_PCT'].mean():.1f}%")
        with col3:
            st.metric("Production Days", len(plot_df[plot_df["BORE_OIL_VOL"] > 0]))
        with col4:
            whp = plot_df["AVG_WHP_P"].replace(0, np.nan).dropna()
            st.metric("Avg WHP", f"{whp.mean():.1f}" if not whp.empty else "N/A")

        # Production chart with anomalies
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Oil & Water Production", "Water Cut %", "Wellhead Pressure"),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3],
        )

        # Oil production
        fig.add_trace(
            go.Scatter(x=plot_df["DATEPRD"], y=plot_df["BORE_OIL_VOL"],
                       name="Oil (Sm3)", line=dict(color="#2E86AB", width=1)),
            row=1, col=1
        )
        # Water production
        fig.add_trace(
            go.Scatter(x=plot_df["DATEPRD"], y=plot_df["BORE_WAT_VOL"],
                       name="Water (Sm3)", line=dict(color="#A23B72", width=1)),
            row=1, col=1
        )
        # Water cut
        fig.add_trace(
            go.Scatter(x=plot_df["DATEPRD"], y=plot_df["WATER_CUT_PCT"],
                       name="Water Cut %", line=dict(color="#F18F01", width=1)),
            row=2, col=1
        )
        # Wellhead pressure
        whp_data = plot_df[plot_df["AVG_WHP_P"] > 0]
        fig.add_trace(
            go.Scatter(x=whp_data["DATEPRD"], y=whp_data["AVG_WHP_P"],
                       name="WHP", line=dict(color="#C73E1D", width=1)),
            row=3, col=1
        )

        # Overlay anomalies
        anomalies = get_anomalies(daily_df, selected_well)
        if not anomalies.empty:
            for _, row in anomalies.iterrows():
                color = {"Critical": "red", "High": "orange", "Medium": "yellow"}.get(
                    str(row.get("SEVERITY", "Medium")), "yellow"
                )
                # Add anomaly marker on appropriate subplot
                if "Oil" in str(row["ANOMALY_TYPE"]):
                    subplot_row = 1
                elif "Water Cut" in str(row["ANOMALY_TYPE"]):
                    subplot_row = 2
                elif "Pressure" in str(row["ANOMALY_TYPE"]):
                    subplot_row = 3
                else:
                    subplot_row = 1

                fig.add_trace(
                    go.Scatter(
                        x=[row["DATEPRD"]], y=[row["VALUE"]],
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="triangle-up"),
                        name=row["ANOMALY_TYPE"],
                        showlegend=False,
                        hovertext=f"{row['ANOMALY_TYPE']}: {row['VALUE']:.1f}",
                    ),
                    row=subplot_row, col=1
                )

        fig.update_layout(height=700, template="plotly_white", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


# --- TAB 2: AI Chat ---
CHAT_HISTORY_MAX_MESSAGES = 24  # cap prior turns to limit tokens (must be even-ish; pairs of user/assistant)


def _session_to_langchain_history(messages: list) -> list:
    """Map stored chat dicts to LangChain messages for the agent (prior turns only)."""
    lc_messages = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not content or not role:
            continue
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    if len(lc_messages) > CHAT_HISTORY_MAX_MESSAGES:
        lc_messages = lc_messages[-CHAT_HISTORY_MAX_MESSAGES:]
    return lc_messages


with tab_chat:
    st.markdown("### 💬 Subsurface AI Assistant")

    # Keep the tab short so chat_input stays near the bottom; details live in an expander
    with st.expander("About this chat & example questions", expanded=False):
        st.markdown(
            "Ask about production, drilling reports, anomalies, decline, or recovery. "
            "Follow-up replies use this conversation (e.g. “use the same well” or “proceed with available data”)."
        )
        st.markdown("""
        - What is the water cut trend for well F-11?
        - Were there any drilling problems reported for well F-12?
        - Calculate the decline rate for F-1 C over the last 12 months
        - What anomalies have been detected across all wells?
        - Based on the completion reports, what formations were encountered in F-14?
        - Compare production performance between F-11 and F-12
        - What is the current recovery factor for F-1 C?
        """)
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f'<div class="source-citation">'
                            f'<strong>{source.get("doc_type", "Document")}</strong> | '
                            f'{source.get("well_name", "Unknown")} | '
                            f'{source.get("source_file", "")}<br>'
                            f'{source.get("excerpt", "")[:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    if prompt := st.chat_input("Ask about the Volve field..."):
        prior_lc = _session_to_langchain_history(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and searching documents..."):
                try:
                    agent = get_agent(daily_df)
                    result = agent.invoke({
                        "input": prompt,
                        "chat_history": prior_lc,
                    })

                    response = result["output"]
                    st.markdown(response)

                    sources = []
                    for step in result.get("intermediate_steps", []):
                        action, observation = step
                        if action.tool == "search_well_documents" and "Source" in str(observation):
                            sources.append({
                                "doc_type": "Well Document",
                                "excerpt": str(observation)[:300],
                            })

                    if sources:
                        with st.expander("📄 Sources"):
                            for source in sources:
                                st.markdown(
                                    f'<div class="source-citation">'
                                    f'<strong>{source.get("doc_type", "Document")}</strong> | '
                                    f'{source.get("well_name", "Unknown")} | '
                                    f'{source.get("source_file", "")}<br>'
                                    f'{source.get("excerpt", "")[:200]}...'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# --- TAB 3: Anomaly Detection ---
with tab_anomalies:
    st.markdown("### ⚠️ Production Anomaly Detection")
    st.markdown("Automated detection of unusual patterns in production data using statistical analysis.")

    well_for_anomaly = st.selectbox(
        "Select well for anomaly analysis",
        options=["All Wells"] + wells,
        key="anomaly_well_select",
    )

    well_filter = None if well_for_anomaly == "All Wells" else well_for_anomaly
    anomalies = get_anomalies(daily_df, well_filter)

    if anomalies.empty:
        st.success("No anomalies detected in the selected data.")
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            critical = len(anomalies[anomalies["SEVERITY"] == "Critical"])
            st.metric("Critical", critical)
        with col2:
            high = len(anomalies[anomalies["SEVERITY"] == "High"])
            st.metric("High", high)
        with col3:
            medium = len(anomalies[anomalies["SEVERITY"] == "Medium"])
            st.metric("Medium", medium)

        # Anomaly type breakdown
        st.markdown("#### Anomaly Distribution")
        type_counts = anomalies["ANOMALY_TYPE"].value_counts().reset_index()
        type_counts.columns = ["Anomaly Type", "Count"]
        fig = px.bar(type_counts, x="Anomaly Type", y="Count",
                     color="Anomaly Type", template="plotly_white")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly timeline
        st.markdown("#### Anomaly Timeline")
        fig = px.scatter(
            anomalies, x="DATEPRD", y="VALUE",
            color="ANOMALY_TYPE", symbol="SEVERITY",
            hover_data=["WELL_NAME", "METRIC"],
            template="plotly_white",
            title="Detected Anomalies Over Time",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Detail table
        st.markdown("#### Anomaly Details")
        display_cols = ["DATEPRD", "WELL_NAME", "ANOMALY_TYPE", "METRIC",
                        "VALUE", "SEVERITY"]
        available_cols = [c for c in display_cols if c in anomalies.columns]
        st.dataframe(
            anomalies[available_cols].sort_values("DATEPRD", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
