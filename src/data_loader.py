"""
Production Data Loader
Loads and cleans the Volve production dataset (daily + monthly).
"""
import pandas as pd
import numpy as np
from src.config import PRODUCTION_DATA_PATH


def load_daily_production(path: str = None) -> pd.DataFrame:
    """Load and clean daily production data."""
    path = path or PRODUCTION_DATA_PATH
    df = pd.read_excel(path, sheet_name="Daily Production Data")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse date
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])

    # Sort by well and date
    df = df.sort_values(["WELL_BORE_CODE", "DATEPRD"]).reset_index(drop=True)

    # Calculate derived metrics
    total_liquid = df["BORE_OIL_VOL"] + df["BORE_WAT_VOL"]
    df["WATER_CUT_PCT"] = np.where(
        total_liquid > 0,
        (df["BORE_WAT_VOL"] / total_liquid) * 100,
        0.0
    )

    # Gas-Oil Ratio (GOR)
    df["GOR"] = np.where(
        df["BORE_OIL_VOL"] > 0,
        df["BORE_GAS_VOL"] / df["BORE_OIL_VOL"],
        0.0
    )

    # Standardize well name for display
    df["WELL_NAME"] = df["WELL_BORE_CODE"].str.replace("NO ", "", regex=False).str.strip()

    return df


def _monthly_oil_water_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Resolve oil / water volume columns (Equinor export variants)."""
    cols = set(df.columns)
    oil = "Oil\nSm3" if "Oil\nSm3" in cols else "Oil" if "Oil" in cols else None
    water = "Water\nSm3" if "Water\nSm3" in cols else "Water" if "Water" in cols else None
    if oil is None or water is None:
        raise KeyError(
            "Monthly sheet must include Oil and Water columns "
            "(either 'Oil' / 'Water' or 'Oil\\nSm3' / 'Water\\nSm3')."
        )
    return oil, water


def load_monthly_production(path: str = None) -> pd.DataFrame:
    """Load and clean monthly production data."""
    path = path or PRODUCTION_DATA_PATH
    df = pd.read_excel(path, sheet_name="Monthly Production Data")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop unit-only / blank rows (some exports put a header row with NaN Year/Month)
    df = df.dropna(subset=["Year", "Month", "Wellbore name"], how="any").copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Year", "Month"])
    df = df[(df["Month"] >= 1) & (df["Month"] <= 12)]

    df["DATE"] = pd.to_datetime(
        {"year": df["Year"].astype(int), "month": df["Month"].astype(int), "day": 1}
    )

    oil_col, water_col = _monthly_oil_water_columns(df)
    oil_vol = df[oil_col].fillna(0).astype(float)
    water_vol = df[water_col].fillna(0).astype(float)
    total_liquid = oil_vol + water_vol
    df["WATER_CUT_PCT"] = np.where(
        total_liquid > 0,
        (water_vol / total_liquid) * 100,
        0.0
    )

    # Sort
    df = df.sort_values(["Wellbore name", "DATE"]).reset_index(drop=True)

    # Standardize well name
    df["WELL_NAME"] = df["Wellbore name"].astype(str).str.strip()

    return df


def get_well_list(df: pd.DataFrame) -> list:
    """Get sorted list of unique well names."""
    if "WELL_NAME" in df.columns:
        return sorted(df["WELL_NAME"].unique().tolist())
    return []


def get_well_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary table for all wells."""
    summary = df.groupby("WELL_NAME").agg(
        first_date=("DATEPRD", "min"),
        last_date=("DATEPRD", "max"),
        total_oil=("BORE_OIL_VOL", "sum"),
        total_water=("BORE_WAT_VOL", "sum"),
        total_gas=("BORE_GAS_VOL", "sum"),
        avg_water_cut=("WATER_CUT_PCT", "mean"),
        production_days=("DATEPRD", "count"),
    ).reset_index()

    summary["first_date"] = summary["first_date"].dt.strftime("%Y-%m-%d")
    summary["last_date"] = summary["last_date"].dt.strftime("%Y-%m-%d")
    summary = summary.round(2)

    return summary


def query_production_data(df: pd.DataFrame, well_name: str = None,
                          start_date: str = None, end_date: str = None,
                          metric: str = None) -> pd.DataFrame:
    """
    Query production data with filters.
    Used by the agent as a tool to answer production-related questions.
    """
    result = df.copy()

    if well_name:
        # Fuzzy match on well name
        mask = result["WELL_NAME"].str.contains(well_name, case=False, na=False)
        result = result[mask]

    if start_date:
        result = result[result["DATEPRD"] >= pd.to_datetime(start_date)]

    if end_date:
        result = result[result["DATEPRD"] <= pd.to_datetime(end_date)]

    if metric and metric in result.columns:
        result = result[["DATEPRD", "WELL_NAME", metric]]

    return result
