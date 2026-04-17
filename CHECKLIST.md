# Project Nomos: Master Checklist

This is your personal checklist. You can tick these off as we complete them!

## Phase 1: Data Architecture & Engineering
- [x] 1.1 Project Initialization & Environment Setup
- [x] 1.2 Multi-Source Data Ingestion (Yahoo + Kite Connect)
- [x] 1.3 Statistical Transformations (Log-Returns)
- [x] 1.4 Feature Engineering (VIX Spreads, Volatility Detrending, Correlations)
- [x] 1.5 Stationarity Diagnostics (ADF/KPSS) & Z-Score Scaling
- [x] 1.6 Advanced Sanitization (Winsorization & Iterative Differencing)

## Phase 2: Regime Detection (HMM)
- [x] 2.1 Hidden Markov Model Architecture (Gaussian HMM)
- [x] 2.2 Model Training Pipeline
- [x] 2.3 Regime Interpretation & State Labeling (Bull, Bear, Neutral)
- [x] 2.4 Transition Stability Analysis

## Phase 3: Volatility Modeling (GJR-GARCH)
- [ ] 3.1 Regime-Specific Variance Estimation
- [ ] 3.2 GJR-GARCH Parameter Calibration
- [ ] 3.3 Volatility Forecast Generation

## Phase 4: Risk Governance & Allocation
- [x] 4.1 Regime-Aware Risk Budgeting
- [x] 4.2 Conditional Value-at-Risk (CVaR) Optimization
- [x] 4.3 Dynamic Capital Allocation Strategy

## Phase 5: Backtesting & Deployment
- [ ] 5.1 Walk-Forward Performance Evaluation
- [ ] 5.2 Transaction Cost & Slippage Analysis
- [ ] 5.3 Live Dashboard & Risk Reporting
