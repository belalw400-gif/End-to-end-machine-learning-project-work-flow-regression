import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from pathlib import Path

# ============================
# MUST SET PAGE CONFIG FIRST
# ============================
st.set_page_config(page_title="Housing Price Prediction", page_icon="🏠", layout="wide")

# Paths (local)
HOLDOUT_ENGINEERED_PATH = "data/processed/holdout_fe.csv"
HOLDOUT_META_PATH = "data/processed/holdout_cleaned.csv"

API_URL = "http://localhost:8000/predict"

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# UI
# ============================

page_style = """
<style>
body, .stApp, .main {
  background: radial-gradient(circle at 20% 10%, rgba(88,91,255,0.16), transparent 30%),
              radial-gradient(circle at 85% 10%, rgba(255,114,214,0.12), transparent 18%),
              linear-gradient(135deg, #020617 0%, #071520 48%, #0a1c34 100%);
  color: #f7f9ff;
}
.stButton>button {
  background: linear-gradient(135deg, #5b6cff, #9d61ff);
  color: #ffffff;
  border: none;
  border-radius: 14px;
  height: 3rem;
  padding: 0.65rem 1.2rem;
  box-shadow: 0 18px 45px rgba(73, 81, 204, 0.28);
}
.stButton>button:hover {
  background: linear-gradient(135deg, #6e7dff, #b470ff);
}
div[data-testid="stHeader"] {
  background: transparent;
}
section[data-testid="stSidebar"] {
  background: rgba(4, 8, 18, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.08);
}
.css-1d391kg, .css-18e3th9 {
  background: transparent;
}
.css-1lcbmhc, .css-1v0mbdj {
  background-color: rgba(8, 14, 28, 0.9) !important;
  border-radius: 18px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="metric-container"] {
  background: rgba(8, 14, 28, 0.92);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

st.markdown(
    """
    <div style='padding: 1.5rem 2rem; border-radius: 24px; background: rgba(10, 18, 40, 0.95); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 40px 80px rgba(0,0,0,0.30);'>
        <h1 style='margin:0; font-size:3rem; letter-spacing:0.03em;'>🏠 Housing Price Prediction — Holdout Explorer</h1>
        <p style='margin:0.75rem 0 0; color:#b1bff1; font-size:1.05rem; line-height:1.7;'>
            Explore model predictions across years, regions, and months.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")

        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

          
            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()

           
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

           
            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            view = view.sort_values("date")
            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

          
            # ============================
            # Yearly Trend Chart
            # ============================
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)

            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient="records")

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_region, index=yearly_data.index).astype(float)

            # Aggregate by month
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(lambda m: "Selected" if m == month else "Other")
            highlight_month = month

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}",
                template="plotly_dark"
            )
            fig.update_traces(mode="lines+markers", marker=dict(size=8), line=dict(width=4))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f4f7ff"),
                legend=dict(bgcolor="rgba(10,16,35,0.8)", bordercolor="rgba(255,255,255,0.12)", borderwidth=1),
                margin=dict(t=60, b=30, l=30, r=30)
            )
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="rgba(255, 99, 132, 0.15)",
                opacity=0.35,
                layer="below",
                line_width=0,
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")