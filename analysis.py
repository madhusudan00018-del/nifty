#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Empirical Analysis of Technical Indicators and Macro Factors on Nifty Equities
A Panel Data Econometrics Study (2014 - 2024)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from nsepython import nse_eq_symbols
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. DATA ACQUISITION & INGESTION
# ---------------------------------------------------------

print("Fetching Nifty 200 constituents list...")
url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
nifty200_df = pd.read_csv(url)
companies = (nifty200_df["Symbol"].astype(str) + ".NS").tolist()

print(f"Total constituents to download: {len(companies)}")

# Fetch historical data from Yahoo Finance
all_data = []
failed = []

for ticker in companies:
    try:
        df = yf.download(
            ticker,
            start="2014-01-01",
            end="2025-01-01",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if df.empty:
            failed.append(ticker)
            continue
            
        df = df.reset_index()
        df["Firm"] = ticker
        
        # Flatten MultiIndex columns if present
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[["Date", "Firm", "Close", "Open", "High", "Low", "Volume"]]
        all_data.append(df)
    except Exception as e:
        failed.append(ticker)

panel_df = pd.concat(all_data, ignore_index=True)
print(f"Successfully downloaded {panel_df['Firm'].nunique()} firms.")

# ---------------------------------------------------------
# 2. DATA CLEANING & SPECIFICATION FILTERS
# ---------------------------------------------------------

# Filter out firms with sparse trading history (keep those with >= 1500 observations)
obs_count = panel_df.groupby("Firm").size()
valid_firms_density = obs_count[obs_count >= 1500].index
panel_df = panel_df[panel_df["Firm"].isin(valid_firms_density)]

# Sort panel logically by Firm and Date
panel_df = panel_df.sort_values(by=["Firm", "Date"]).reset_index(drop=True)

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------

# Log Daily Returns
panel_df["Log_Return"] = panel_df.groupby("Firm")["Close"].transform(
    lambda x: np.log(x / x.shift(1))
)

# Rolling Volatility (21-day window)
panel_df["Volatility"] = panel_df.groupby("Firm")["Log_Return"].transform(
    lambda x: x.rolling(21).std()
)

# Rolling Momentum (21-day window)
panel_df["Momentum"] = panel_df.groupby("Firm")["Close"].transform(
    lambda x: (x / x.shift(21)) - 1
)

# Price to Moving Average Ratio (21-day)
ma21 = panel_df.groupby("Firm")["Close"].transform(
    lambda x: x.rolling(21).mean()
)
panel_df["MA_Ratio"] = panel_df["Close"] / ma21

# Custom Relative Strength Index (RSI-14)
delta = panel_df.groupby("Firm")["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(panel_df["Firm"]).transform(lambda x: x.rolling(14).mean())
avg_loss = loss.groupby(panel_df["Firm"]).transform(lambda x: x.rolling(14).mean())

rs = avg_gain / avg_loss
panel_df["RSI"] = 100 - (100 / (1 + rs))

# ---------------------------------------------------------
# 4. OUTLIER MITIGATION (WINSORIZATION)
# ---------------------------------------------------------

# Winsorize key variables at 1% and 99% to reduce the impact of extreme price events
for col in ["Log_Return", "Momentum", "Volatility"]:
    lower_bound = panel_df[col].quantile(0.01)
    upper_bound = panel_df[col].quantile(0.99)
    panel_df[col] = np.clip(panel_df[col], lower_bound, upper_bound)

# ---------------------------------------------------------
# 5. MACRO CONTROLS
# ---------------------------------------------------------

# Fetch Nifty 50 Index as Market Proxy
nifty_idx = yf.download("^NSEI", start="2014-01-01", end="2025-01-01", interval="1d", auto_adjust=True, progress=False)
nifty_idx = nifty_idx.reset_index()
nifty_idx.columns = [col[0] if isinstance(col, tuple) else col for col in nifty_idx.columns]
nifty_idx["Market_Return"] = np.log(nifty_idx["Close"] / nifty_idx["Close"].shift(1))
nifty_idx = nifty_idx[["Date", "Market_Return"]]

# Merge Market Return control
panel_df = panel_df.merge(nifty_idx, on="Date", how="left")

# Fetch India VIX as Systemic Risk Proxy
vix_idx = yf.download("^INDIAVIX", start="2014-01-01", end="2025-01-01", interval="1d", auto_adjust=True, progress=False)
vix_idx = vix_idx.reset_index()
vix_idx.columns = [col[0] if isinstance(col, tuple) else col for col in vix_idx.columns]
vix_idx = vix_idx[["Date", "Close"]].rename(columns={"Close": "VIX"})

# Merge VIX control
panel_df = panel_df.merge(vix_idx, on="Date", how="left")

# Final Drop of NaNs resulting from rolling windows/shifts
panel_df = panel_df.dropna()

# Apply strict data coverage threshold (keep firms with at least 85% of total trading days)
total_dates = panel_df["Date"].nunique()
obs_per_firm = panel_df.groupby("Firm").size()
coverage_ratio = obs_per_firm / total_dates
valid_firms_coverage = coverage_ratio[coverage_ratio >= 0.85].index
panel_df = panel_df[panel_df["Firm"].isin(valid_firms_coverage)]

print(f"Final panel contains {panel_df['Firm'].nunique()} firms and {len(panel_df)} total observations.")

# ---------------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS & PLOTS
# ---------------------------------------------------------

os.makedirs("plots", exist_ok=True)

# 1. Observations per Firm (Sample Coverage)
plt.figure(figsize=(12, 6))
panel_df.groupby("Firm").size().sort_values().plot(kind="bar")
plt.title("Observations Per Firm (Sample Coverage)")
plt.ylabel("Number of Observations")
plt.xlabel("Firm")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/sample_coverage.png", dpi=300)
plt.close()

# 2. Correlation Matrix
corr_matrix = panel_df[["Log_Return", "Volatility", "Momentum", "MA_Ratio", "RSI", "Market_Return", "VIX"]].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Variable Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=300)
plt.close()

# 3. Log Return Distribution
plt.figure(figsize=(12, 6))
plt.hist(panel_df["Log_Return"], bins=100, edgecolor='black', alpha=0.7)
plt.title("Distribution of Daily Log Returns (Winsorized)")
plt.xlabel("Log Return")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/returns_distribution.png", dpi=300)
plt.close()

# 4. Market Volatility over Time
plt.figure(figsize=(14, 6))
panel_df.groupby("Date")["Volatility"].mean().plot()
plt.title("Average Equity Volatility Over Time (2014-2024)")
plt.xlabel("Date")
plt.ylabel("Average Volatility")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/average_volatility.png", dpi=300)
plt.close()

# 5. Average Daily Return Over Time
plt.figure(figsize=(14, 6))
panel_df.groupby("Date")["Log_Return"].mean().plot()
plt.title("Average Daily Returns Across Portfolio")
plt.xlabel("Date")
plt.ylabel("Average Return")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/average_returns.png", dpi=300)
plt.close()

# 6. Sector Mapping & Visualizations
sector_map = {
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking", "AXISBANK.NS": "Banking", "KOTAKBANK.NS": "Banking",
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", "TECHM.NS": "IT",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "POWERGRID.NS": "Energy"
}

# Assign sectors (default to 'Other')
panel_df["Sector"] = panel_df["Firm"].map(sector_map).fillna("Other")

# Returns by Sector
plt.figure(figsize=(10, 5))
panel_df.groupby("Sector")["Log_Return"].mean().sort_values().plot(kind="bar")
plt.title("Average Daily Returns by Sector")
plt.ylabel("Average Daily Return")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/sector_returns.png", dpi=300)
plt.close()

# Volatility by Sector
plt.figure(figsize=(10, 5))
panel_df.groupby("Sector")["Volatility"].mean().sort_values().plot(kind="bar")
plt.title("Average Volatility by Sector")
plt.ylabel("Average Volatility")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/sector_volatility.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 7. PANEL ECONOMETRIC REGRESSIONS
# ---------------------------------------------------------

# Set standard multi-index for panel (Firm, Date)
panel_df["Date"] = pd.to_datetime(panel_df["Date"])
panel_data = panel_df.set_index(["Firm", "Date"])

exog_vars = ["Volatility", "Momentum", "MA_Ratio", "RSI", "Market_Return", "VIX"]
X = sm.add_constant(panel_data[exog_vars])
y = panel_data["Log_Return"]

print("\n--- 1. Pooled OLS Regression ---")
pooled_model = PanelOLS(y, X, entity_effects=False, time_effects=False)
pooled_results = pooled_model.fit()
print(pooled_results.summary.tables[1])

print("\n--- 2. Fixed Effects (Entity Effects) Panel Regression ---")
fe_model = PanelOLS(y, X, entity_effects=True, time_effects=False)
fe_results = fe_model.fit()
print(fe_results.summary.tables[1])

print("\n--- 3. Random Effects Panel Regression ---")
re_model = RandomEffects(y, X)
re_results = re_model.fit()
print(re_results.summary.tables[1])

print("\n--- 4. First Difference Panel Regression ---")
# Build first differences for variables to handle non-stationarity
fd_df = panel_data[["Log_Return"] + exog_vars].groupby(level=0).diff().dropna()
y_fd = fd_df["Log_Return"]
X_fd = sm.add_constant(fd_df[exog_vars])

fd_model = PanelOLS(y_fd, X_fd, entity_effects=False, time_effects=False)
fd_results = fd_model.fit()
print(fd_results.summary.tables[1])

print("\nAnalysis pipeline complete. All plots saved to the 'plots/' folder.")
