import pandas as pd
import numpy as np
from typing import List, Optional

class DataProcessor:
    """
    Handles statistical transformations and feature engineering for Project Nomos.
    """
    
    def __init__(self, config: dict):
        self.config = config

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
