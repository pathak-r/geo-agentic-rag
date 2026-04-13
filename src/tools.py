"""
Agent Tools
Functions that the LLM agent can call to answer questions.
"""
import pandas as pd
import numpy as np
from typing import Optional
from src.data_loader import query_production_data
from src.anomaly import detect_anomalies, get_anomaly_summary
from src.vector_store import search_documents_multi_query


def production_query_tool(df: pd.DataFrame, well_name: str,
                          metric: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> str:
    """
    Query production data for a specific well.
    Returns a text summary of the data.
    """
    result = query_production_data(df, well_name, start_date, end_date, metric)

    if result.empty:
        return f"No production data found for well matching '{well_name}'."

    well = result["WELL_NAME"].iloc[0]
    date_range = f"{result['DATEPRD'].min().strftime('%Y-%m-%d')} to {result['DATEPRD'].max().strftime('%Y-%m-%d')}"

    # Generate summary statistics
    summary_lines = [
        f"Production data for {well} ({date_range}):",
        f"  Total records: {len(result)}",
    ]

    if "BORE_OIL_VOL" in result.columns:
        summary_lines.append(f"  Oil production: avg={result['BORE_OIL_VOL'].mean():.1f}, "
                             f"max={result['BORE_OIL_VOL'].max():.1f}, "
                             f"total={result['BORE_OIL_VOL'].sum():.1f} Sm3")
    if "BORE_WAT_VOL" in result.columns:
        summary_lines.append(f"  Water production: avg={result['BORE_WAT_VOL'].mean():.1f}, "
                             f"max={result['BORE_WAT_VOL'].max():.1f}, "
                             f"total={result['BORE_WAT_VOL'].sum():.1f} Sm3")
    if "WATER_CUT_PCT" in result.columns:
        summary_lines.append(f"  Water cut: avg={result['WATER_CUT_PCT'].mean():.1f}%, "
                             f"max={result['WATER_CUT_PCT'].max():.1f}%")
    if "BORE_GAS_VOL" in result.columns:
        summary_lines.append(f"  Gas production: total={result['BORE_GAS_VOL'].sum():.1f} Sm3")
    if "AVG_WHP_P" in result.columns:
        whp = result["AVG_WHP_P"].replace(0, np.nan).dropna()
        if not whp.empty:
            summary_lines.append(f"  Wellhead pressure: avg={whp.mean():.1f}, "
                                 f"min={whp.min():.1f}, max={whp.max():.1f}")

    # Recent trend (last 30 days of data)
    recent = result.tail(30)
    if len(recent) > 1 and "BORE_OIL_VOL" in recent.columns:
        first_half = recent.head(15)["BORE_OIL_VOL"].mean()
        second_half = recent.tail(15)["BORE_OIL_VOL"].mean()
        if first_half > 0:
            change = ((second_half - first_half) / first_half) * 100
            trend = "increasing" if change > 5 else "decreasing" if change < -5 else "stable"
            summary_lines.append(f"  Recent trend: {trend} ({change:+.1f}% over last 30 records)")

    return "\n".join(summary_lines)


def anomaly_check_tool(df: pd.DataFrame, well_name: Optional[str] = None) -> str:
    """
    Check for anomalies in production data.
    Returns natural language description of detected anomalies.
    """
    anomalies = detect_anomalies(df, well_name)
    return get_anomaly_summary(anomalies)


def calculate_recovery_factor(df: pd.DataFrame, well_name: str,
                              ooip_sm3: float = None) -> str:
    """
    Calculate recovery factor for a well.
    RF = Cumulative Oil Produced / Original Oil in Place (OOIP)
    If OOIP not provided, returns cumulative production only.
    """
    mask = df["WELL_NAME"].str.contains(well_name, case=False, na=False)
    well_data = df[mask]

    if well_data.empty:
        return f"No data found for well matching '{well_name}'."

    cumulative_oil = well_data["BORE_OIL_VOL"].sum()
    cumulative_water = well_data["BORE_WAT_VOL"].sum()
    cumulative_gas = well_data["BORE_GAS_VOL"].sum()

    result = [
        f"Cumulative production for {well_data['WELL_NAME'].iloc[0]}:",
        f"  Oil: {cumulative_oil:,.0f} Sm3",
        f"  Water: {cumulative_water:,.0f} Sm3",
        f"  Gas: {cumulative_gas:,.0f} Sm3",
    ]

    if ooip_sm3:
        rf = (cumulative_oil / ooip_sm3) * 100
        result.append(f"  Recovery Factor: {rf:.2f}% (based on OOIP of {ooip_sm3:,.0f} Sm3)")
    else:
        result.append("  Recovery Factor: Cannot calculate without OOIP estimate.")
        result.append("  Note: The Volve field total OOIP was approximately 12.6 million Sm3.")

    return "\n".join(result)


def calculate_decline_rate(df: pd.DataFrame, well_name: str,
                           period_months: int = 6) -> str:
    """
    Calculate production decline rate for a well.
    Uses exponential decline model: q(t) = qi * exp(-D*t)
    """
    mask = df["WELL_NAME"].str.contains(well_name, case=False, na=False)
    well_data = df[mask].copy()

    if well_data.empty:
        return f"No data found for well matching '{well_name}'."

    well_data = well_data[well_data["BORE_OIL_VOL"] > 0].sort_values("DATEPRD")

    if len(well_data) < 30:
        return f"Insufficient production data for decline analysis (need 30+ producing days)."

    # Get the last N months of data
    end_date = well_data["DATEPRD"].max()
    start_date = end_date - pd.Timedelta(days=period_months * 30)
    period_data = well_data[well_data["DATEPRD"] >= start_date]

    if len(period_data) < 10:
        return f"Insufficient data in the last {period_months} months for decline analysis."

    # Simple decline rate: (initial rate - final rate) / initial rate / time
    initial_rate = period_data.head(10)["BORE_OIL_VOL"].mean()
    final_rate = period_data.tail(10)["BORE_OIL_VOL"].mean()

    if initial_rate <= 0:
        return "Cannot calculate decline rate: initial production rate is zero."

    decline_pct = ((initial_rate - final_rate) / initial_rate) * 100
    monthly_decline = decline_pct / period_months

    result = [
        f"Decline analysis for {well_data['WELL_NAME'].iloc[0]} "
        f"(last {period_months} months):",
        f"  Initial avg rate: {initial_rate:.1f} Sm3/day",
        f"  Current avg rate: {final_rate:.1f} Sm3/day",
        f"  Total decline: {decline_pct:.1f}%",
        f"  Monthly decline rate: {monthly_decline:.1f}%/month",
    ]

    if decline_pct > 30:
        result.append("  ⚠ Significant decline — intervention may be warranted.")
    elif decline_pct > 10:
        result.append("  Moderate decline — within normal range for mature wells.")
    elif decline_pct < 0:
        result.append("  Production is increasing (negative decline).")
    else:
        result.append("  Mild decline — well performing within expectations.")

    return "\n".join(result)


def document_search_tool(query: str, embeddings_model) -> str:
    """
    Search well documents (drilling reports, completion reports) for relevant information.
    Returns formatted search results with source citations.
    """
    results = search_documents_multi_query(query, embeddings_model)

    if not results:
        return "No relevant documents found for this query."

    formatted = ["Relevant documents found:\n"]
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        source = meta.get("source_file", "Unknown")
        well = meta.get("well_name", "Unknown")
        doc_type = meta.get("doc_type", "Unknown").replace("_", " ").title()
        score = result["score"]

        formatted.append(f"--- Source {i} [{doc_type} | {well}] (relevance: {score:.2f}) ---")
        formatted.append(result["text"][:1200])  # Limit text length
        formatted.append("")

    return "\n".join(formatted)
