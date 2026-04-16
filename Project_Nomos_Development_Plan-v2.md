# Project Nomos: Regime-Adaptive Multi-Asset Systematic Allocator

**Objective:** To architect an industrialized, cross-asset systematic trading framework that utilizes probabilistic regime detection and asymmetric volatility targeting to manage a 'Trinity' of NIFTY50, Gold, and USDINR spot markets.

---

## 1. Data Architecture & Pre-Processing
To ensure statistical integrity and satisfy the requirements of Tier 1 quantitative firms, the data layer implements rigorous stationarity transformations on spot market data.

### Asset Trinity (Spot Markets)
- **Equity:** NIFTY50 Index Spot (Primary Alpha Source).
- **Commodity:** Gold Spot (Defensive/Non-Correlated Proxy).
- **Currency:** USDINR Spot (Macro-Regime Indicator).

### Pipeline Flow
1. **Ingestion:** Raw spot price, volume, and IndiaVIX tick data.
2. **Transformation:** - Conversion of all spot price series to **Log-Returns**.
    - **Feature Engineering:** IndiaVIX Spread (VIX - 20d MA), Detrended Volume (Volume / 10d Avg), and Cross-Asset Rolling Correlations.
3. **Statistical Validation:**
    - **ADF (Augmented Dickey-Fuller):** Null hypothesis rejection to ensure no unit root.
    - **KPSS:** Confirmation of stationarity around a deterministic trend.
4. **Normalization:** Application of **Z-Score Scaling** (Mean 0, Std Dev 1) to all features to ensure equal weighting in the state-space model.

---

## 2. Probabilistic Regime Detection (The Brain)
Project Nomos treats market conditions as 'Hidden States' that cannot be observed directly but can be inferred from the Trinity’s spot returns.

### Hidden Markov Model (HMM)
- **Engine:** Gaussian Hidden Markov Model.
- **Latent States ($N=3$):** - *State 0 (Quiet Bull):* Low volatility, positive drift.
    - *State 1 (Mean-Reverting Sideways):* High noise, zero drift.
    - *State 2 (Volatile Bear):* High volatility, negative drift, high cross-asset correlation.
- **Decision Logic:** The HMM outputs the posterior probability of the current state and the transition matrix (the probability of shifting regimes).

---

## 3. Asymmetric Volatility Targeting
Standardizing risk across different market regimes via a 'Nomos' (Law-based) scaling approach.

### Volatility Estimation: GJR-GARCH
- Unlike standard GARCH, the **GJR-GARCH(1,1)** model captures the 'Leverage Effect' (the tendency for volatility to spike more aggressively on the downside in NIFTY50 spot).
- **Formula:** $\sigma_{t}^2 = \omega + (lpha + \gamma I_{t-1})\epsilon_{t-1}^2 + eta\sigma_{t-1}^2$
- **Half-Life EWMA:** A secondary memory-based estimator used to smooth the realized volatility input.

### Position Sizing
- **Target Volatility:** Fixed at an institutional 15% (annualized).
- **Scaling Factor:** $Weight = rac{\sigma_{target}}{\sigma_{GJR-GARCH}}$
- **Outcome:** Automatic deleveraging during high-volatility regime shifts detected by the HMM.

---

## 4. Institutional Risk Management
### Regime-Aware Expected Shortfall (CVaR)
- **Concept:** Tail risk is non-stationary.
- **Process:** 1. Filter historical spot data based on the HMM's currently identified state.
    2. Calculate CVaR at the 97.5% confidence level using state-conditional returns.
    3. **Stop-Loss Logic:** If the current portfolio CVaR exceeds the regime-specific threshold, the system triggers a 'Hard Exit' or aggressive Delta-hedging.

---

## 5. Execution Micro-Architecture
Utilizing the **Adapter Pattern** for production-grade reliability, bridging spot signals to derivative execution.

- **Signal Listener:** Receives regime-aware target weights derived from spot analytics.
- **Execution Handler:** Analyzes **Limit Order Book (LOB)** dynamics and order-flow imbalance in the corresponding futures/options markets.
- **Liquidity Logic:** Employs passive limit orders during 'Quiet Bull' regimes and aggressive market-taking during 'Volatile Bear' transitions to minimize slippage.

---

## 6. Project Impact & JD Mapping
This framework is designed to counter **96% of the JD Pool** requirements:
- **Man Group / Jump Trading:** Satisfied via LOB analytics and GJR-GARCH volatility modeling.
- **Jane Street / QRT:** Satisfied via HMM-based feature engineering and experimental design.
- **Barclays / ICE:** Satisfied via CVaR-based risk governance and cross-asset derivatives logic.
