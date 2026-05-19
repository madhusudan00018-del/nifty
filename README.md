# 📈 Nifty Quantitative Portfolio & Panel Econometric Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-Document-green.svg?logo=latex&logoColor=white)](https://www.latex-project.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced quantitative finance and panel data econometrics pipeline analyzing the top Indian firms listed on the **Nifty 100/200 indices**. This project fetches over a decade of daily financial market data (2014 – 2025), engineers high-frequency technical indicators, performs sector-wise risk-return profiling, and estimates multi-variable Panel Data Regressions (Fixed Effects, Random Effects, and Pooled OLS).

The final report is generated as an academic LaTeX document (`nifty100.tex`) complete with embedded mathematical equations and high-quality visualizations.

---

## 🌟 Key Features

* **Automated Data Pipeline:** Fetches real-time equity symbols and metadata from the National Stock Exchange of India (NSE) via `nsepython` and downloads historical daily pricing data via `yfinance`.
* **Advanced Feature Engineering:**
  * **Log Returns:** Logarithmic transformation of daily price ratios.
  * **Rolling Volatility:** Standard deviation of returns over a rolling window.
  * **Momentum:** One-year rolling asset returns.
  * **Moving Average (MA) Ratio:** Asset price relative to its moving average.
  * **Relative Strength Index (RSI):** Technical momentum oscillator.
  * **Macro Controls:** Dynamic incorporation of the India VIX (Volatility Index) and overall market returns.
* **Panel Data Econometrics:** Estimates Pooled OLS, Fixed Effects, and Random Effects panel models to examine stock returns against endogenous technical variables and exogenous market volatility.
* **Academic LaTeX Report:** An elegantly typeset document (`nifty100.tex`) summarizing findings, regression tables, and correlation heatmaps.

---

## 📁 Repository Structure

```
├── nifty100.tex         # Academic LaTeX paper & notebook export
├── plots/               # High-resolution generated charts & visualizations
│   ├── output_34_0.png  # Observations per firm (data completeness)
│   ├── output_42_0.png  # Asset correlation matrix heatmap
│   ├── output_43_0.png  # Distribution of daily log returns
│   ├── output_44_0.png  # Average market volatility over time
│   ├── output_45_0.png  # Average daily returns across firms
│   ├── output_50_0.png  # Average daily returns by sector
│   └── output_51_0.png  # Average volatility by sector
├── .gitignore           # Smart LaTeX and Python build artifact exclusions
└── README.md            # Highly descriptive project documentation
```

---

## 📊 Visualizations Gallery

### 1. Data Completeness & Observations
Tracks the density of data points across the sample of Indian equities from 2014 to 2025, ensuring statistical significance.
![Observations Per Firm](plots/output_34_0.png)

### 2. Feature Correlation Heatmap
Explores linear relationships among returns, volatility, VIX, RSI, momentum, and the moving average ratio.
![Correlation Matrix](plots/output_42_0.png)

### 3. Market Volatility & Returns Over Time
Highlights regime shifts, macro shocks (such as the 2020 crash), and market corrections.
| Market Volatility Over Time | Daily Returns Distribution |
|---|---|
| ![Average Market Volatility](plots/output_44_0.png) | ![Distribution of Log Returns](plots/output_43_0.png) |

### 4. Sector Risk-Return Profiles
Contrasts defensive sectors (like FMCG and Pharma) against high-beta cyclical sectors (such as Auto, Banking, and Energy).
| Returns by Sector | Volatility by Sector |
|---|---|
| ![Returns by Sector](plots/output_50_0.png) | ![Volatility by Sector](plots/output_51_0.png) |

---

## 🛠️ Prerequisites & Setup

### 1. Python Dependencies
Ensure you have Python installed. You can install all necessary packages via `pip`:

```bash
pip install pandas numpy yfinance nsepython seaborn matplotlib statsmodels linearmodels
```

### 2. Compiling the LaTeX Report
To compile the academic report into a PDF, you will need a LaTeX distribution (like MacTeX or TeX Live). Compile using `pdflatex` or `xelatex`:

```bash
pdflatex nifty100.tex
```

---

## ✍️ Authors & License
Developed as a quantitative research project. Under the [MIT License](LICENSE).
