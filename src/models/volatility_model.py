import numpy as np
import pandas as pd
from arch import arch_model
import joblib
import os
from typing import Dict, List, Optional

class NomosVolatilityModel:
    """
    Volatility Modeling Engine for Project Nomos.
    Implements Regime-Specific Variance and GJR-GARCH forecasting.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model_result = None
        self.regime_stats = {}

    def get_regime_stats(self, df: pd.DataFrame, return_col: str, regime_col: str) -> pd.DataFrame:
        """
        Step 3.1: Calculate annualized volatility and mean returns per regime.
        """
        if return_col not in df.columns or regime_col not in df.columns:
            raise KeyError(f"Columns {return_col} or {regime_col} missing from DataFrame.")
            
        # Group by regime
        stats = df.groupby(regime_col)[return_col].agg(['mean', 'std', 'count'])
        
        # Annualize (assuming 252 trading days)
        stats['annualized_return'] = stats['mean'] * 252
        stats['annualized_vol'] = stats['std'] * np.sqrt(252)
        
        self.regime_stats = stats.to_dict(orient='index')
        return stats

    def fit_gjr_garch(self, returns: pd.Series):
        """
        Step 3.2: Fit GJR-GARCH(1,1) model.
        """
        # Fit on raw log-returns (e.g. 0.005)
        # We use rescale=True to let arch handle scaling internally for convergence
        model = arch_model(returns, p=1, q=1, o=1, vol='Garch', dist='normal', rescale=True)
        self.model_result = model.fit(disp='off')
        
        return self.model_result

    def get_conditional_volatility(self) -> pd.Series:
        """
        Retrieve the full series of in-sample conditional volatility.
        Annualized and de-scaled to original space.
        """
        if self.model_result is None:
            raise ValueError("Model must be fitted before retrieving volatility.")
            
        # Divide by the internal scale to get back to original decimal space
        vol_series = (self.model_result.conditional_volatility / self.model_result.scale) * np.sqrt(252)
        return vol_series

    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Step 3.3: Generate 1-step ahead annualized volatility forecast.
        """
        if self.model_result is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        forecast = self.model_result.forecast(horizon=horizon)
        # Variance is in scaled space. To de-scale: variance / (scale^2)
        last_variance = forecast.variance.iloc[-1].values[0]
        # De-scale the standard deviation: sqrt(variance) / scale
        forecasted_vol = (np.sqrt(last_variance) / self.model_result.scale) * np.sqrt(252)
        
        return forecasted_vol

    def save_model(self, path: str):
        if self.model_result:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model_result, path)
            print(f"Volatility model saved to {path}")

    @staticmethod
    def load_model(path: str) -> 'NomosVolatilityModel':
        return joblib.load(path)
