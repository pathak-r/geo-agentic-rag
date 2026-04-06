"""
Anomaly Detection for Well Production Data
Detects anomalies in water cut, production rates, and pressure readings.
"""
import pandas as pd
import numpy as np
from src.config import ANOMALY_WINDOW, ANOMALY_THRESHOLD


def detect_anomalies(df: pd.DataFrame, well_name: str = None) -> pd.DataFrame:
    """
    Run anomaly detection across all metrics for one or all wells.
    Returns DataFrame with anomaly flags and descriptions.
    """
    if well_name:
        mask = df["WELL_NAME"].str.contains(well_name, case=False, na=False)
        data = df[mask].copy()
    else:
        data = df.copy()

    if data.empty:
        return data

    all_anomalies = []

    for well in data["WELL_NAME"].unique():
        well_data = data[data["WELL_NAME"] == well].copy()
        well_data = well_data.sort_values("DATEPRD")

        # Only analyze wells with enough data points
        if len(well_data) < ANOMALY_WINDOW + 5:
            continue

        # Detect anomalies for each metric
        water_cut_anomalies = _detect_metric_anomaly(
            well_data, "WATER_CUT_PCT", "Water Cut Spike"
        )
        oil_drop_anomalies = _detect_production_drop(
            well_data, "BORE_OIL_VOL", "Oil Production Drop"
        )
        pressure_anomalies = _detect_metric_anomaly(
            well_data, "AVG_WHP_P", "Wellhead Pressure Anomaly"
        )
        gor_anomalies = _detect_metric_anomaly(
            well_data, "GOR", "Gas-Oil Ratio Spike"
        )

        for anomaly_df in [water_cut_anomalies, oil_drop_anomalies,
                           pressure_anomalies, gor_anomalies]:
            if not anomaly_df.empty:
                all_anomalies.append(anomaly_df)

    if not all_anomalies:
        return pd.DataFrame(columns=[
            "DATEPRD", "WELL_NAME", "ANOMALY_TYPE", "METRIC",
            "VALUE", "EXPECTED_RANGE_LOW", "EXPECTED_RANGE_HIGH", "SEVERITY"
        ])

    return pd.concat(all_anomalies, ignore_index=True)


def _detect_metric_anomaly(well_data: pd.DataFrame, metric: str,
                           anomaly_type: str) -> pd.DataFrame:
    """Detect anomalies using rolling z-score method."""
    if metric not in well_data.columns:
        return pd.DataFrame()

    series = well_data[metric].fillna(0)

    # Skip if no meaningful data
    if series.max() == 0:
        return pd.DataFrame()

    rolling_mean = series.rolling(window=ANOMALY_WINDOW, min_periods=10).mean()
    rolling_std = series.rolling(window=ANOMALY_WINDOW, min_periods=10).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z_scores = (series - rolling_mean) / rolling_std

    # Flag anomalies where z-score exceeds threshold
    anomaly_mask = z_scores.abs() > ANOMALY_THRESHOLD

    if not anomaly_mask.any():
        return pd.DataFrame()

    anomalies = well_data[anomaly_mask][["DATEPRD", "WELL_NAME"]].copy()
    anomalies["ANOMALY_TYPE"] = anomaly_type
    anomalies["METRIC"] = metric
    anomalies["VALUE"] = series[anomaly_mask].values
    anomalies["EXPECTED_RANGE_LOW"] = (rolling_mean - ANOMALY_THRESHOLD * rolling_std)[anomaly_mask].values
    anomalies["EXPECTED_RANGE_HIGH"] = (rolling_mean + ANOMALY_THRESHOLD * rolling_std)[anomaly_mask].values

    # Severity based on z-score magnitude
    z_vals = z_scores[anomaly_mask].abs().values
    anomalies["SEVERITY"] = pd.cut(
        z_vals,
        bins=[0, 2.5, 3.5, float("inf")],
        labels=["Medium", "High", "Critical"]
    )

    return anomalies


def _detect_production_drop(well_data: pd.DataFrame, metric: str,
                            anomaly_type: str) -> pd.DataFrame:
    """Detect sudden production drops (negative anomalies only)."""
    if metric not in well_data.columns:
        return pd.DataFrame()

    series = well_data[metric].fillna(0)

    if series.max() == 0:
        return pd.DataFrame()

    rolling_mean = series.rolling(window=ANOMALY_WINDOW, min_periods=10).mean()
    rolling_std = series.rolling(window=ANOMALY_WINDOW, min_periods=10).std()
    rolling_std = rolling_std.replace(0, np.nan)

    z_scores = (series - rolling_mean) / rolling_std

    # Only flag negative anomalies (drops)
    anomaly_mask = z_scores < -ANOMALY_THRESHOLD

    if not anomaly_mask.any():
        return pd.DataFrame()

    anomalies = well_data[anomaly_mask][["DATEPRD", "WELL_NAME"]].copy()
    anomalies["ANOMALY_TYPE"] = anomaly_type
    anomalies["METRIC"] = metric
    anomalies["VALUE"] = series[anomaly_mask].values
    anomalies["EXPECTED_RANGE_LOW"] = (rolling_mean - ANOMALY_THRESHOLD * rolling_std)[anomaly_mask].values
    anomalies["EXPECTED_RANGE_HIGH"] = (rolling_mean + ANOMALY_THRESHOLD * rolling_std)[anomaly_mask].values

    z_vals = z_scores[anomaly_mask].abs().values
    anomalies["SEVERITY"] = pd.cut(
        z_vals,
        bins=[0, 2.5, 3.5, float("inf")],
        labels=["Medium", "High", "Critical"]
    )

    return anomalies


def get_anomaly_summary(anomalies: pd.DataFrame) -> str:
    """Generate a natural language summary of detected anomalies for the agent."""
    if anomalies.empty:
        return "No anomalies detected in the production data."

    lines = []
    for well in anomalies["WELL_NAME"].unique():
        well_anomalies = anomalies[anomalies["WELL_NAME"] == well]
        counts = well_anomalies["ANOMALY_TYPE"].value_counts()
        severity_counts = well_anomalies["SEVERITY"].value_counts()

        line = f"Well {well}: {len(well_anomalies)} anomalies detected"
        details = [f"{count} {atype}" for atype, count in counts.items()]
        line += f" ({', '.join(details)})"

        critical = severity_counts.get("Critical", 0)
        if critical > 0:
            line += f" — {critical} CRITICAL"

        lines.append(line)

    return "\n".join(lines)
