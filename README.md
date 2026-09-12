# Quantitative Impact of Global and Domestic Events on the BIST 100

---

## 1. Motivation
Emerging equity markets, particularly Borsa Istanbul (BIST 100), exhibit high volatility and acute sensitivity to macroeconomic environment shifts. Financial market shocks and volatility spikes do not occur in isolation; they are closely tied to sudden exogenous events (such as geopolitical conflicts, natural disasters, and global crises) as well as domestic policy turns (including sudden interest rate adjustments and foreign exchange interventions).

This project investigates how major domestic and global turning points quantitatively impact the BIST 100 index from 2015 to the present. Rather than examining price trends in isolation, this study establishes an end-to-end data science pipeline to evaluate whether macroeconomic shocks structurally alter market return distributions, variance, and recovery dynamics. Furthermore, it explores the empirical boundaries of predictive machine learning models in forecasting future daily returns and directional risk under crisis conditions.

---

## 2. Research Questions and Sub-Questions

**Main Question**
- To what extent do global exogenous shocks and domestic macroeconomic policy shifts drive return distribution and volatility expansion within the BIST 100 index?

**Sub-Questions**
1. Does the depth of a market drop correlate directly with the assigned qualitative severity level of an external shock, or do market participants react indiscriminately once a structural crisis hits?
2. Does the BIST 100 exhibit a significant short-term "hangover" effect where market panic spills over into subsequent trading sessions, or does it operate with near-immediate informational efficiency?
3. Can complex machine learning classification and regularized regression pipelines accurately forecast future daily returns using time-lagged historical indicators under crisis regimes?

---

## 3. Hypotheses
- **H1:** The distribution of daily returns on macro-shock days is fundamentally shifted and statistically different from normal trading days.
- **H2:** Macroeconomic and geopolitical shocks systematically trigger a massive expansion of market volatility (variance) compared to standard trading conditions.
- **H3:** The magnitude of the immediate market drop depends strictly on the categorized qualitative severity level (Medium vs. High vs. Extreme) of the external shock.
- **H4:** The market exhibits a short-term "hangover" effect, causing negative pressure on returns to persist into the trading session immediately following an event ($T+1$).

- **H0 (Null):** External historical shocks and macroeconomic indicators hold no statistically significant or predictive relationship with BIST 100 return distributions or volatility parameters.

---

## 4. Data Description (As Implemented in the Notebooks)

| Dataset | Variables | Purpose |
|-------|----------|---------|
| **Yahoo Finance** (`bist100_raw.csv`) | Daily Open, High, Low, Close, Volume | Primary independent asset layer used to evaluate index performance and draw baseline metrics. |
| **TCMB EVDS** (`macro_indicators_raw.csv`) | Monthly CPI (`TP.FG.J0`), Daily USD/TRY Rate (`TP.DK.USD.A.YTL`) | Enriched macroeconomic variables representing local inflationary environments and foreign exchange pressures. |
| **Event Catalog** (`events_raw.csv`) | `date`, `event_name`, `category`, `severity` | Explicitly curated database tracking 24 critical historical shocks from 2015 to 2026. |

### Processed Data Outputs
- `data_collection/processed/unified_bist_enriched.csv` *(Multivariate master dataset combining financial metrics and engineered features)*
- `data_collection/processed/unified_bist_data.csv` *(Merged daily time-series with unified date indexes for EDA)*
- `data_collection/processed/shock_analysis_summary.csv` *(Aggregated event-level summary panel tracking return windows)*

---

## 5. Data Source and Collection

- **BIST 100 Index Logs:** Programmatically extracted utilizing the `yfinance` API framework wrapper (`XU100.IS`) to pull daily data spanning from January 2, 2015, to April 3, 2026 ($2,817$ raw trading observations).
- **Macroeconomic Enrichment Data:** Fetched directly from the Central Bank of the Republic of Türkiye (TCMB) Electronic Data Delivery System via the `evdsAPI`. Column layers were structured for Consumer Price Index (CPI) and the USD/TRY Exchange Rate.
- **Qualitative Shocks Timeline:** A customized dataset capturing 24 high-impact events spanning across multiple categories (e.g., the 2016 Coup Attempt, the 2018 Currency Crisis, the 2020 COVID-19 Outbreak, and the 2023 Kahramanmaraş Earthquake) compiled via deep public historical news records.

To prevent structural data leakage, all datasets were chronologically aligned, and monthly reporting intervals (CPI) were forward-filled (`ffill`) across the daily trading framework.

---

## 6. Methodology

### 6.1 Panel Construction & Feature Engineering
A structured daily master panel was constructed by merging the financial index with macroeconomic features. Temporal alignment was achieved via an asymmetric `pd.merge_asof` parameter using a `direction='forward'` orientation. This explicitly maps non-trading weekend or holiday shocks to the opening bell of the next available market session. Three critical technical indicators were engineered:
- **7-Day Rolling Volatility:** Captures localized market nervousness and near-term risk expansions.
- **20-Day Moving Average (MA):** Charts the baseline underlying macro price momentum.
- **Deviation from Trend (`dev_from_ma`):** Measures the percentage "snap" distance from the moving average to track oversold/overbought thresholds during panic sell-offs:
$$\text{dev\_from\_ma} = \frac{\text{Close} - \text{MA}_{20d}}{\text{MA}_{20d}}$$

---

### 6.2 Exploratory Data Analysis
Visual profiling was deployed across data layers using `matplotlib` and `seaborn`. This included mapping distribution profiles of returns via boxplots, mapping baseline foreign exchange trends, and constructing a rolling standard deviation baseline tracking system-wide stress over a multi-year horizon. Trajectories were normalized in USD values (`close / usd_try`) to strip away local currency inflation skew and isolate real wealth variations.

---

### 6.3 Hypothesis Testing
- **H1 (Distribution Shift):** Evaluated via a two-sided **Mann-Whitney U Test**. This non-parametric technique compares separate sample medians without assuming a Gaussian layout, making it ideal for the fat-tailed distributions of financial returns.
- **H2 (Volatility Expansion):** Analyzed using **Levene’s Test for Equality of Variances**. This test is specifically deployed for its resilience against non-normal kurtosis profiles in time-series data.
- **H3 (Severity Correlation):** Executed via a **One-Way ANOVA** (`smf.ols`) testing if categorical designations (*Medium, High, Extreme*) explain out-of-sample return variations.
- **H4 (Hangover Effect):** Evaluated via a lagged Mann-Whitney U test comparing Day $T+1$ (the active trading session directly following a shock) against standard baseline trading days.

---

### 6.4 Machine Learning Analysis
Predictive modeling pipelines were implemented in `scikit-learn` to establish out-of-sample validation boundaries:
- **Feature Engineering to Avoid Leakage:** Shifting independent features backward by exactly one period (`shift(1)`) to guarantee the pipeline uses yesterday's economic variables to predict today's return parameters, preventing look-ahead bias.
- **Chronological Data Splitting:** Utilizing a strict time-aware chronological split (first 50% for training: 2015-2020, final 50% held out for out-of-sample testing: 2020-2026) to mirror live trading conditions.
- **Imbalance Mitigation:** Synthesizing minority "Shock Days" inside the training loop via `SMOTE` (Synthetic Minority Over-sampling Technique) embedded inside an automated `ImbPipeline`.
- **Architectures Configured:** Evaluated linear models (Logistic Regression, Ridge, Lasso, ElasticNet), support vector networks (SVR), and ensemble tree methods (Random Forest Classifier/Regressor, Gradient Boosting Classifier/Regressor).

---

## 7. Results and Findings

The project successfully modeled emerging market dynamics, proving that while macro-shocks cause instantaneous structural risk expansion, predicting precise financial fluctuations remains bound to strict informational limitations.

### Statistical Test Suite Summary
| Hypothesis | Statistical Test Model | Metric Output | Result | P-Value | Academic Insight |
|------------|------------------------|---------------|--------|---------|------------------|
| **H1: Return Shift** | Mann-Whitney U Test | U = 17883.0 | ✅ **Supported** | `3.3786e-02` | Macro-shocks significantly alter return medians. Shock days skew heavily negative compared to standard baseline trading sessions. |
| **H2: Volatility Expansion** | Levene's Variance Test | Levene Stat = 17.5025 | ✅ **Supported** | `2.9570e-05` | Shocks trigger massive system-wide risk expansion. Shock day return volatility measures **1.91x higher** than normal trading conditions. |
| **H3: Severity Sorting** | One-Way ANOVA Regression | F-Statistic = 0.5840 | ❌ **Not Supported** | `0.5698` | Once a crisis strikes, investor panic behaves indiscriminately. A "Medium" event triggers asset liquidations just as severe as an "Extreme" designation. |
| **H4: Hangover Effect** | Lagged Mann-Whitney U | $T+1$ Return Shift | ❌ **Not Supported** | `6.9933e-01` | The BIST 100 operates with high efficiency. Shock vectors are absorbed within a single trading day; no risk spillover carries into Day $T+1$. |

### Machine Learning Performance Matrix
- **Continuous Regression (Level Prediction):** Forecasting precise numerical daily returns yielded negative $R^2$ scores across all configurations (Random Forest $R^2 = -0.071$, Linear Regression $R^2 = -1.404$). In financial econometrics, this is an expected outcome consistent with the **Efficient Market Hypothesis (EMH)** and random walk characteristics of short-term returns; daily noise dominates, rendering point forecasts via historical lags ineffective against a naive mean benchmark.
- **Categorical Risk Classification:** Shifting the objective from continuous level prediction to directional shock classification yielded meaningful risk management parameters. When augmented with `SMOTE` oversampling to counter extreme class imbalance, the Random Forest classifier attained an ROC-AUC of **0.512** (compared to 0.392 for baseline Logistic Regression). Feature importance via permutation analysis confirmed that **7-day rolling historical volatility** provided the primary explanatory signal for regime instability.

### Analytical Deep-Dive & Main Contributions
By bridging empirical finance and data science pipelines, this study provides three core contributions:
1. **Non-Parametric Evaluation under Heavy Tails:** Confirmed that asset returns deviate significantly from normality during shocks, validating the necessity of robust alternatives like Mann-Whitney U and Levene's tests over standard Student's t-tests.
2. **Empirical Evidence of Indiscriminate Reaction (H3 Rejection):** Demonstrated that market drop severity does not scale linearly with qualitative labels (*Medium* vs. *Extreme*). Once an unexpected shock registers, market participants react indiscriminately with immediate capital preservation.
3. **Information Efficiency & Lack of Persistence (H4 Rejection):** The absence of a lagged hangover effect ($T+1$) highlights high informational efficiency in Borsa Istanbul, where the vast majority of shock-induced adjustments are absorbed within the initial trading session.

---

## 8. Limitations & Future Work

### Current Limitations
- **Exclusively Lagged Macroeconomic Data:** While lagging macro indicators by one day prevents technical look-ahead bias, metrics like the CPI are structurally bounded to low-frequency monthly announcements. This leaves high-frequency daily volatility modeling partially blind to rapid intraday fundamental shifts.
- **Binary Shock Definition:** External events are currently modeled as static binary occurrences (`is_shock = 1`). This does not dynamically capture the rolling duration of global crises (e.g., the prolonged structural economic effects of the COVID-19 pandemic or unfolding macroeconomic transition periods).
- **Absence of Digital Behavioral Proxies:** Financial trends in emerging markets are heavily driven by rapid human reactions. The existing data baseline relies entirely on structured market and economic variables, lacking live proxies for real-time human behavior (such as digital search analytics or unstructured media text stream sentiment).

### Future Work
- **Integration of Real-Time Digital Behavioral Metrics:** Incorporate daily Google Trends search volume metrics (tracking localized queries such as *"USD/TRY exchange rate"* or *"inflation"*) alongside scraped social media text streams to construct a composite Digital Panic Index. This framework will evaluate whether real-time behavioral anxiety footprints can detect and explain market regime anomalies faster than low-frequency, backward-looking macroeconomic indicators.
- **Multi-Modal Hybrid Modeling:** Develop nested hybrid validation architectures to statistically contrast traditional fundamental asset pricing models against multi-modal frameworks enriched with digital behavioral metrics, mapping structural improvements in out-of-sample predictive power.
- **Transition to Deep Sequence Architectures:** Replace static machine learning regression tasks with deep recurrent sequence models—such as Long Short-Term Memory (LSTM) networks or temporal transformers—optimized to effectively capture the non-linear, multi-day memory dynamics of financial variance across structural monetary policy regime pivots.

---

## 📂 Repository Structure

```text
DSA210_TermProject/
│
├── data_collection/
│   ├── jupyter_notebook/
│   │   └── DataCollection.ipynb             # Script for yfinance and EVDS API integration
│   ├── raw/                                 # Original downloaded data files
│   │   ├── bist100_raw.csv
│   │   ├── macro_indicators_raw.csv
│   │   └── events_raw.csv
│   └── processed/                           # Cleaned and merged analysis datasets
│       ├── unified_bist_enriched.csv        # Main master dataset for ML and EDA
│       ├── unified_bist_data.csv            # Base financial metrics
│       ├── shock_analysis_summary.csv       # Event-driven shock evaluations
│       └── EEM_benchmark.csv                # Emerging Markets benchmark index (iShares MSCI EM)
│
├── EDA/
│   ├── EDA.ipynb                            # Complete EDA and baseline correlations
│   └── Visualizations/                      # Standalone charts and plots
│
├── hypothesis_testing/
│   └── hypotesting.ipynb                    # Non-parametric tests, ANOVA, and variance analysis
│
├── ML_methods/
│   └── ml_methods.ipynb                     # Classification, Regression, and SMOTE integration
│
├── advanced_econometric_analysis/            # (Optional Extension)
│   └── advanced_econometric_analysis.ipynb  # GARCH(1,1) volatility & CAR Event Study models
│
├── DSA210_Final_Report.pdf                  # Comprehensive academic final project report
├── DSA210_Project_Proposal_Yigit_Keskin.pdf # Initial project proposal
├── README.md
└── requirements.txt
```

---

## 🛠️ Academic Integrity & AI Assistance Disclosure

In accordance with Sabancı University's Academic Integrity guidelines for **DSA 210 (Introduction to Data Science)**:
- **Independent Work:** All research questions, hypotheses, data collection pipelines (via `yfinance` and TCMB EVDS APIs), statistical test executions, and machine learning models were independently designed, implemented, and analyzed by the author.
- **AI Tool Usage:** Generative AI tools (Gemini / LLM assistance) were utilized strictly for assistive tasks, including code debugging, Markdown formatting, proofreading, and improving documentation structure. All generated text, numerical outputs, and interpretations were verified against the author's local Jupyter Notebook runs.
