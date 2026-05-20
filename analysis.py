#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nifty Quantitative Portfolio & Panel Econometric Analysis
Automatically extracted and cleaned from nifty100.tex
"""

# =============================================================================
# [CELL 1]
# =============================================================================
companies = [

    # Banking & Financials
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "INDUSINDBK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",

    # IT
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "LTIM.NS",

    # Energy & Oil
    "RELIANCE.NS",
    "ONGC.NS",
    "BPCL.NS",
    "IOC.NS",
    "GAIL.NS",

    # FMCG
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",
    "DABUR.NS",
    "GODREJCP.NS",

    # Pharma
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "DIVISLAB.NS",
    "CIPLA.NS",
    "LUPIN.NS",
    "APOLLOHOSP.NS",

    # Auto
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "M&M.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",

    # Infrastructure / Industrials
    "LT.NS",
    "ULTRACEMCO.NS",
    "ASIANPAINT.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "GRASIM.NS",
    "SHREECEM.NS",

    # Telecom & Utilities
    "BHARTIARTL.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "ADANIPORTS.NS",

    # Consumer & Retail
    "TITAN.NS",
    "TRENT.NS",
    "PIDILITIND.NS",

    # Chemicals / Others
    "UPL.NS",
    "ADANIENT.NS",
    "COALINDIA.NS",
    "BEL.NS"

]

print("Total Firms:", len(companies))

# =============================================================================
# [CELL 2]
# =============================================================================
# pip install nsepython

# =============================================================================
# [CELL 3]
# =============================================================================
from nsepython import *
import pandas as pd

# =============================================================================
# [CELL 5]
# =============================================================================
nifty200 = pd.DataFrame(nse_eq_symbols())

print(nifty200.head())
print("Total Symbols:", len(nifty200))

# =============================================================================
# [CELL 6]
# =============================================================================
url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

nifty200 = pd.read_csv(url)

print(nifty200.head())

print("Total Firms:", len(nifty200))

# =============================================================================
# [CELL 7]
# =============================================================================
companies = (
    nifty200["Symbol"]
    .astype(str)
    + ".NS"
).tolist()

print(companies[:10])

print("Total Companies:", len(companies))

# =============================================================================
# [CELL 8]
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np

# =============================================================================
# [CELL 9]
# =============================================================================
all_data = []

failed = []

for ticker in companies:

    print(f"Downloading: {ticker}")

    try:

        df = yf.download(
            ticker,
            start="2014-01-01",
            end="2025-01-01",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        # Empty check
        if df.empty:
            failed.append(ticker)
            continue

        # Reset index
        df = df.reset_index()

        # Add ticker
        df["Firm"] = ticker

        # Keep important columns
        df = df[
            [
                "Date",
                "Firm",
                "Close",
                "Open",
                "High",
                "Low",
                "Volume"
            ]
        ]

        all_data.append(df)

    except Exception as e:

        print("FAILED:", ticker)
        print(e)

        failed.append(ticker)

# =============================================================================
# [CELL 10]
# =============================================================================
panel_df = pd.concat(all_data, ignore_index=True)

print(panel_df.shape)

# =============================================================================
# [CELL 11]
# =============================================================================
print("Successful Firms:", panel_df["Firm"].nunique())

print("\nFailed Firms:")
print(failed)

# =============================================================================
# [CELL 12]
# =============================================================================
obs_check = (
    panel_df
    .groupby("Firm")
    .size()
    .sort_values()
)

print(obs_check.head(20))

# =============================================================================
# [CELL 13]
# =============================================================================
cleaned_data = []

for df in all_data:

    # Flatten MultiIndex columns
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    # Keep needed columns only
    df = df[
        [
            "Date",
            "Firm",
            "Close",
            "Open",
            "High",
            "Low",
            "Volume"
        ]
    ]

    cleaned_data.append(df)

# Rebuild panel
panel_df = pd.concat(
    cleaned_data,
    ignore_index=True
)

print(panel_df.shape)

panel_df.head()

# =============================================================================
# [CELL 14]
# =============================================================================
obs_check = (
    panel_df
    .groupby("Firm")
    .size()
)

valid_firms = obs_check[
    obs_check >= 1500
].index

panel_df = panel_df[
    panel_df["Firm"].isin(valid_firms)
]

print(panel_df.shape)

print("Remaining Firms:",
      panel_df["Firm"].nunique())

# =============================================================================
# [CELL 15]
# =============================================================================
panel_df = panel_df.sort_values(
    by=["Firm", "Date"]
)

panel_df = panel_df.reset_index(drop=True)

panel_df.head()

# =============================================================================
# [CELL 16]
# =============================================================================
panel_df["Log_Return"] = (
    panel_df
    .groupby("Firm")["Close"]
    .transform(
        lambda x: np.log(
            x / x.shift(1)
        )
    )
)

# =============================================================================
# [CELL 17]
# =============================================================================
panel_df["Volatility"] = (
    panel_df
    .groupby("Firm")["Log_Return"]
    .transform(
        lambda x: x.rolling(21).std()
    )
)

# =============================================================================
# [CELL 18]
# =============================================================================
panel_df["Momentum"] = (
    panel_df
    .groupby("Firm")["Close"]
    .transform(
        lambda x: (
            x / x.shift(21)
        ) - 1
    )
)

# =============================================================================
# [CELL 19]
# =============================================================================
ma21 = (
    panel_df
    .groupby("Firm")["Close"]
    .transform(
        lambda x: x.rolling(21).mean()
    )
)

panel_df["MA_Ratio"] = (
    panel_df["Close"] / ma21
)

# =============================================================================
# [CELL 20]
# =============================================================================
delta = (
    panel_df
    .groupby("Firm")["Close"]
    .diff()
)

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = (
    gain.groupby(panel_df["Firm"])
    .transform(
        lambda x: x.rolling(14).mean()
    )
)

avg_loss = (
    loss.groupby(panel_df["Firm"])
    .transform(
        lambda x: x.rolling(14).mean()
    )
)

rs = avg_gain / avg_loss

panel_df["RSI"] = (
    100 - (100 / (1 + rs))
)

# =============================================================================
# [CELL 21]
# =============================================================================
panel_df[
    [
        "Log_Return",
        "Volatility",
        "Momentum",
        "MA_Ratio",
        "RSI"
    ]
].describe()

# =============================================================================
# [CELL 22]
# =============================================================================
lower_ret = panel_df["Log_Return"].quantile(0.01)

upper_ret = panel_df["Log_Return"].quantile(0.99)

panel_df["Log_Return"] = np.clip(
    panel_df["Log_Return"],
    lower_ret,
    upper_ret
)

# =============================================================================
# [CELL 23]
# =============================================================================
lower_mom = panel_df["Momentum"].quantile(0.01)

upper_mom = panel_df["Momentum"].quantile(0.99)

panel_df["Momentum"] = np.clip(
    panel_df["Momentum"],
    lower_mom,
    upper_mom
)

# =============================================================================
# [CELL 24]
# =============================================================================
lower_vol = panel_df["Volatility"].quantile(0.01)

upper_vol = panel_df["Volatility"].quantile(0.99)

panel_df["Volatility"] = np.clip(
    panel_df["Volatility"],
    lower_vol,
    upper_vol
)

# =============================================================================
# [CELL 25]
# =============================================================================
panel_df[
    [
        "Log_Return",
        "Volatility",
        "Momentum",
        "MA_Ratio",
        "RSI"
    ]
].describe()

# =============================================================================
# [CELL 26]
# =============================================================================
nifty = yf.download(
    "^NSEI",
    start="2014-01-01",
    end="2025-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# =============================================================================
# [CELL 27]
# =============================================================================
nifty = nifty.reset_index()

nifty.columns = [
    col[0] if isinstance(col, tuple) else col
    for col in nifty.columns
]

# =============================================================================
# [CELL 28]
# =============================================================================
nifty["Market_Return"] = np.log(
    nifty["Close"] / nifty["Close"].shift(1)
)

nifty = nifty[
    [
        "Date",
        "Market_Return"
    ]
]

nifty.head()

# =============================================================================
# [CELL 29]
# =============================================================================
panel_df = panel_df.merge(
    nifty,
    on="Date",
    how="left"
)

panel_df.head()

# =============================================================================
# [CELL 30]
# =============================================================================
vix = yf.download(
    "^INDIAVIX",
    start="2014-01-01",
    end="2025-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# =============================================================================
# [CELL 31]
# =============================================================================
vix = vix.reset_index()

vix.columns = [
    col[0] if isinstance(col, tuple) else col
    for col in vix.columns
]

# =============================================================================
# [CELL 32]
# =============================================================================
vix = vix[
    [
        "Date",
        "Close"
    ]
]

vix = vix.rename(
    columns={
        "Close": "VIX"
    }
)

vix.head()

# =============================================================================
# [CELL 33]
# =============================================================================
panel_df = panel_df.merge(
    vix,
    on="Date",
    how="left"
)

panel_df.head()

# =============================================================================
# [CELL 34]
# =============================================================================
panel_df = panel_df.dropna()

print(panel_df.shape)

panel_df.head()

# =============================================================================
# [CELL 35]
# =============================================================================
obs_per_firm = (
    panel_df
    .groupby("Firm")
    .size()
)

print(obs_per_firm.describe())

# =============================================================================
# [CELL 36]
# =============================================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

obs_per_firm.sort_values().plot(kind="bar")

plt.title("Observations Per Firm")

plt.ylabel("Number of Observations")

plt.xlabel("Firm")

plt.grid(True)

plt.show()

# =============================================================================
# [CELL 37]
# =============================================================================
date_coverage = panel_df.groupby("Firm")[
    "Date"
].agg(
    ["min", "max", "count"]
)

date_coverage.head(20)

# =============================================================================
# [CELL 38]
# =============================================================================
total_dates = panel_df["Date"].nunique()

print("Total Trading Dates:", total_dates)

coverage_ratio = (
    obs_per_firm / total_dates
)

print(coverage_ratio.describe())

# =============================================================================
# [CELL 39]
# =============================================================================
total_dates = panel_df["Date"].nunique()

obs_per_firm = (
    panel_df
    .groupby("Firm")
    .size()
)

coverage_ratio = (
    obs_per_firm / total_dates
)

coverage_ratio.describe()

# =============================================================================
# [CELL 40]
# =============================================================================
valid_firms = coverage_ratio[
    coverage_ratio >= 0.85
].index

panel_df = panel_df[
    panel_df["Firm"].isin(valid_firms)
]

print(panel_df.shape)

print("Remaining Firms:",
      panel_df["Firm"].nunique())

# =============================================================================
# [CELL 41]
# =============================================================================
print(panel_df.info())

# =============================================================================
# [CELL 42]
# =============================================================================
summary_stats = panel_df[
    [
        "Log_Return",
        "Volatility",
        "Momentum",
        "MA_Ratio",
        "RSI",
        "Market_Return",
        "VIX"
    ]
].describe()

summary_stats

# =============================================================================
# [CELL 43]
# =============================================================================
corr_matrix = panel_df[
    [
        "Log_Return",
        "Volatility",
        "Momentum",
        "MA_Ratio",
        "RSI",
        "Market_Return",
        "VIX"
    ]
].corr()

corr_matrix

# =============================================================================
# [CELL 44]
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,7))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()

# =============================================================================
# [CELL 45]
# =============================================================================
plt.figure(figsize=(12,6))

plt.hist(
    panel_df["Log_Return"],
    bins=100
)

plt.title("Distribution of Daily Log Returns")

plt.xlabel("Log Return")

plt.ylabel("Frequency")

plt.show()

# =============================================================================
# [CELL 46]
# =============================================================================
avg_vol = (
    panel_df
    .groupby("Date")["Volatility"]
    .mean()
)

plt.figure(figsize=(14,6))

plt.plot(
    avg_vol.index,
    avg_vol.values
)

plt.title("Average Market Volatility Over Time")

plt.xlabel("Date")

plt.ylabel("Average Volatility")

plt.grid(True)

plt.show()

# =============================================================================
# [CELL 47]
# =============================================================================
avg_return = (
    panel_df
    .groupby("Date")["Log_Return"]
    .mean()
)

plt.figure(figsize=(14,6))

plt.plot(
    avg_return.index,
    avg_return.values
)

plt.title("Average Daily Returns Across Firms")

plt.xlabel("Date")

plt.ylabel("Average Return")

plt.grid(True)

plt.show()

# =============================================================================
# [CELL 48]
# =============================================================================
sector_map = {

    # Banking
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "KOTAKBANK.NS": "Banking",

    # IT
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",
    "TECHM.NS": "IT",

    # FMCG
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",

    # Pharma
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma",

    # Auto
    "MARUTI.NS": "Auto",
    "TATAMOTORS.NS": "Auto",
    "M&M.NS": "Auto",

    # Energy
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "BPCL.NS": "Energy"

}

# =============================================================================
# [CELL 49]
# =============================================================================
panel_df["Sector"] = (
    panel_df["Firm"]
    .map(sector_map)
)

panel_df["Sector"] = (
    panel_df["Sector"]
    .fillna("Other")
)

panel_df["Sector"].value_counts()

# =============================================================================
# [CELL 50]
# =============================================================================
sector_summary = panel_df.groupby("Sector")[
    [
        "Log_Return",
        "Volatility",
        "Momentum"
    ]
].mean()

sector_summary

# =============================================================================
# [CELL 51]
# =============================================================================
sector_risk = panel_df.groupby("Sector")[
    [
        "Log_Return",
        "Volatility"
    ]
].std()

sector_risk

# =============================================================================
# [CELL 52]
# =============================================================================
sector_returns = (
    panel_df.groupby("Sector")["Log_Return"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10,5))

sector_returns.plot(kind="bar")

plt.title("Average Daily Returns by Sector")

plt.ylabel("Average Daily Return")

plt.grid(True)

plt.show()

# =============================================================================
# [CELL 53]
# =============================================================================
sector_vol = (
    panel_df.groupby("Sector")["Volatility"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10,5))

sector_vol.plot(kind="bar")

plt.title("Average Volatility by Sector")

plt.ylabel("Average Volatility")

plt.grid(True)

plt.show()

# =============================================================================
# [CELL 54]
# =============================================================================
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

# Convert date
panel_df["Date"] = pd.to_datetime(panel_df["Date"])

# Set panel index
panel_df = panel_df.set_index(["Firm", "Date"])

# Check structure
print(panel_df.index)

# Dimensions
print("Observations:", panel_df.shape[0])
print("Variables:", panel_df.shape[1])

# Firms and time periods
print("Number of Firms:", panel_df.index.get_level_values(0).nunique())
print("Number of Dates:", panel_df.index.get_level_values(1).nunique())

# =============================================================================
# [CELL 55]
# =============================================================================
obs_per_firm = panel_df.groupby(level=0).size()

print(obs_per_firm.describe())

# =============================================================================
# [CELL 56]
# =============================================================================
# Define dependent and independent variables

y = panel_df["Log_Return"]

X = panel_df[
    [
        "Volatility",
        "Momentum",
        "Market_Return",
        "VIX"
    ]
]

# Add constant
X = sm.add_constant(X)

# Run pooled OLS
pooled_model = sm.OLS(y, X).fit()

# Print results
print(pooled_model.summary())

# =============================================================================
# [CELL 57]
# =============================================================================
# Fixed Effects Model

fe_model = PanelOLS(
    dependent=panel_df["Log_Return"],
    exog=panel_df[
        [
            "Volatility",
            "Momentum",
            "Market_Return",
            "VIX"
        ]
    ],
    entity_effects=True
)

fe_result = fe_model.fit(
    cov_type="driscoll-kraay"
)

print(fe_result)

# =============================================================================
# [CELL 58]
# =============================================================================
# Random Effects Model

re_model = RandomEffects(
    dependent=panel_df["Log_Return"],
    exog=panel_df[
        [
            "Volatility",
            "Momentum",
            "Market_Return",
            "VIX"
        ]
    ]
)

re_result = re_model.fit()

print(re_result)

# =============================================================================
# [CELL 59]
# =============================================================================
from scipy import stats
import numpy as np

# FE coefficients
b_FE = fe_result.params

# RE coefficients
b_RE = re_result.params

# Difference
b_diff = b_FE - b_RE

# Covariance matrices
cov_FE = fe_result.cov
cov_RE = re_result.cov

# Difference in covariance
cov_diff = cov_FE - cov_RE

# Hausman statistic
hausman_stat = np.dot(
    np.dot(
        b_diff.T,
        np.linalg.inv(cov_diff)
    ),
    b_diff
)

# Degrees of freedom
df = len(b_diff)

# P-value
p_value = 1 - stats.chi2.cdf(
    hausman_stat,
    df
)

print("Hausman Test Statistic:",
      hausman_stat)

print("Degrees of Freedom:",
      df)

print("P-value:",
      p_value)

# =============================================================================
# [CELL 60]
# =============================================================================
from statsmodels.stats.diagnostic import het_breuschpagan

# Residuals
residuals = fe_result.resids

# Explanatory variables
exog = panel_df[
    [
        "Volatility",
        "Momentum",
        "Market_Return",
        "VIX"
    ]
]

# Add constant
exog = sm.add_constant(exog)

# Breusch-Pagan Test
bp_test = het_breuschpagan(
    residuals,
    exog
)

labels = [
    "LM Statistic",
    "LM-Test p-value",
    "F-Statistic",
    "F-Test p-value"
]

print(dict(zip(labels, bp_test)))

# =============================================================================
# [CELL 61]
# =============================================================================
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)

print("Durbin-Watson Statistic:", dw)

# =============================================================================
# [CELL 62]
# =============================================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create VIF dataframe
vif_data = pd.DataFrame()

vif_data["Variable"] = exog.columns

vif_data["VIF"] = [
    variance_inflation_factor(
        exog.values,
        i
    )
    for i in range(exog.shape[1])
]

print(vif_data)

# =============================================================================
# [CELL 63]
# =============================================================================
# Residual matrix
resid_df = residuals.unstack(level=0)

# Correlation matrix
corr_matrix = resid_df.corr()

# Remove diagonal
np.fill_diagonal(corr_matrix.values, np.nan)

# Average correlation
avg_corr = np.nanmean(corr_matrix.values)

print("Average Cross-Sectional Correlation:",
      avg_corr)

# =============================================================================
# [CELL 65]
# =============================================================================
twfe_model = PanelOLS(
    dependent=panel_df["Log_Return"],
    exog=panel_df[
        [
            "Volatility",
            "Momentum"
        ]
    ],
    entity_effects=True,
    time_effects=True
)

twfe_result = twfe_model.fit(
    cov_type="driscoll-kraay"
)

print(twfe_result)

# =============================================================================
# [CELL 66]
# =============================================================================
# Create differenced variables

fd_df = panel_df.copy()

fd_df["D_Log_Return"] = (
    fd_df.groupby(level=0)["Log_Return"]
    .diff()
)

fd_df["D_Volatility"] = (
    fd_df.groupby(level=0)["Volatility"]
    .diff()
)

fd_df["D_Momentum"] = (
    fd_df.groupby(level=0)["Momentum"]
    .diff()
)

# Drop NA after differencing
fd_df = fd_df.dropna()

# Define variables
y_fd = fd_df["D_Log_Return"]

X_fd = fd_df[
    [
        "D_Volatility",
        "D_Momentum"
    ]
]

# Add constant
X_fd = sm.add_constant(X_fd)

# Run FD regression
fd_model = sm.OLS(
    y_fd,
    X_fd
).fit()

print(fd_model.summary())

# =============================================================================
# [CELL 67]
# =============================================================================
panel_df["Lag_Return"] = (
    panel_df.groupby(level=0)["Log_Return"]
    .shift(1)
)

# =============================================================================
# [CELL 68]
# =============================================================================
dynamic_df = panel_df.dropna()

# =============================================================================
# [CELL 69]
# =============================================================================
dynamic_model = PanelOLS(
    dependent=dynamic_df["Log_Return"],
    exog=dynamic_df[
        [
            "Lag_Return",
            "Momentum",
            "Volatility"
        ]
    ],
    entity_effects=True,
    time_effects=True
)

dynamic_result = dynamic_model.fit(
    cov_type="driscoll-kraay"
)

print(dynamic_result)

