import pandas as pd
import numpy as np
from typing import List, Optional
from statsmodels.tsa.stattools import adfuller, kpss

class DataProcessor:
    """
    Handles statistical transformations and feature engineering for Project Nomos.
    """
    
    def __init__(self, config: dict):
        self.config = config

    def check_stationarity(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Run ADF and KPSS tests for a set of columns and return a summary report.
        ADF: Null = Non-Stationary (Want p < 0.05)
        KPSS: Null = Stationary (Want p > 0.05)
        """
        results = []
        for col in columns:
            if col in df.columns:
                series = df[col].dropna()
                # ADF Test
                adf_res = adfuller(series)
                # KPSS Test
                kpss_res = kpss(series, regression='c', nlags='auto')
                
                results.append({
                    "Feature": col,
                    "ADF_p": round(adf_res[1], 4),
                    "KPSS_p": round(kpss_res[1], 4),
                    "Stationary": "YES" if (adf_res[1] < 0.05 and kpss_res[1] > 0.05) else "NO"
                })
        
        report_df = pd.DataFrame(results)
        return report_df

    def enforce_stationarity(self, df: pd.DataFrame, columns: List[str], max_diff: int = 1) -> pd.DataFrame:
        """
        Ensures columns pass both ADF (p < 0.05) and KPSS (p > 0.05).
        Applies:
        1. Winsorization (Outlier cleaning @ 3sigma)
        2. Iterative Differencing (up to max_diff times)
        """
        df_clean = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            # Step 1: Outlier Cleaning (Winsorization at 3 sigma)
            # This often fixes KPSS failures caused by structural spikes
            mean, std = series.mean(), series.std()
            series = series.clip(lower=mean - 3*std, upper=mean + 3*std)
            
            # Step 2: Iterative Differencing
            d = 0
            while d <= max_diff:
                # Test
                adf_p = adfuller(series)[1]
                kpss_p = kpss(series, regression='c', nlags='auto')[1]
                
                if adf_p < 0.05 and kpss_p > 0.05:
                    # Stationary!
                    break
                else:
                    # Difference the series
                    series = series.diff().dropna()
                    d += 1
            
            # Step 3: Update the DataFrame
            # Note: We may lose rows due to differencing, but for HMM we align at the end
            df_clean[col] = series
            
        return df_clean

    def apply_zscore_scaling(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply Z-Score Scaling: (x - mean) / std.
        Ensures all features are on the same scale for the HMM.
        """
        df_scaled = df.copy()
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df_scaled[col] = (df[col] - mean) / std
                else:
                    df_scaled[col] = 0.0
        return df_scaled

    def compute_log_returns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Compute log-returns for the specified columns.
        Formula: r_t = ln(P_t) - ln(P_{t-1})
        """
        df_returns = df.copy()
        
        for col in columns:
            if col in df.columns:
                # Use np.log for natural logarithm
                # Series.diff() on log values gives the log return
                df_returns[f"{col}_Ret"] = np.log(df[col]).diff()
            else:
                print(f"Warning: Column {col} not found for log-return calculation.")
                
        # Drop the first row which will be NaN after diff()
        df_returns.dropna(subset=[f"{col}_Ret" for col in columns if col in df.columns], inplace=True)
        
        return df_returns
    def compute_vix_spread(self, df: pd.DataFrame, vix_col: str, window: int = 20) -> pd.DataFrame:
        """
        Compute VIX Spread: Current VIX - SMA(window).
        Highlights volatility expansion/contraction.
        """
        df_processed = df.copy()
        if vix_col in df.columns:
            sma = df[vix_col].rolling(window=window).mean()
            df_processed["VIX_Spread"] = df[vix_col] - sma
        return df_processed

    def compute_detrended_volume(self, df: pd.DataFrame, vol_col: str, window: int = 10) -> pd.DataFrame:
        """
        Compute Detrended Volume: Volume / SMA(window).
        Highlights unusual liquidity/activity shocks.
        """
        df_processed = df.copy()
        if vol_col in df.columns:
            # We use a moving average to detrend
            ma = df[vol_col].rolling(window=window).mean()
            # Avoid division by zero
            df_processed["Detrended_Vol"] = df[vol_col] / (ma.replace(0, np.nan))
        return df_processed

    def compute_rolling_correlations(self, df: pd.DataFrame, anchor_col: str, target_cols: List[str], window: int = 20) -> pd.DataFrame:
        """
        Compute rolling cross-asset correlations between returns.
        Captures regime-dependent correlation spikes.
        """
        df_processed = df.copy()
        for target in target_cols:
            if anchor_col in df.columns and target in df.columns:
                corr_key = f"Corr_{anchor_col.replace('_Ret', '')}_{target.replace('_Ret', '')}"
                df_processed[corr_key] = df[anchor_col].rolling(window=window).corr(df[target])
        return df_processed
