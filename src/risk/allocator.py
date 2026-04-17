import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os

class NomosAllocator:
    """
    Capital Allocation Engine for Project Nomos.
    Implements Regime-Aware Volatility Targeting and CVaR Risk Governance.
    """

    def __init__(self, config: dict):
        self.config = config
        self.risk_params = config['parameters']['risk']
        self.budgets = self.risk_params['budgets']
        self.max_leverage = self.risk_params.get('max_leverage', 1.0)
        self.cvar_confidence = self.risk_params.get('cvar_confidence', 0.975)

    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Step 4.2: Calculate Conditional Value-at-Risk (Expected Shortfall).
        It measures the average loss in the worst (1-confidence)% of cases.
        """
        conf = confidence if confidence else self.cvar_confidence
        # Sort returns and find the var (Value at Risk)
        sorted_rets = np.sort(returns)
        var_index = int((1 - conf) * len(sorted_rets))
        
        if var_index == 0:
            return np.mean(sorted_rets[:1]) # Fallback for small samples
            
        # CVaR is the average of returns below the VaR threshold
        cvar = np.mean(sorted_rets[:var_index])
        return abs(cvar) # Return as positive number representing risk

    def get_regime_target_vol(self, regime: str) -> float:
        """
        Step 4.1: Map regime labels to Target Volatility budgets.
        """
        return self.budgets.get(regime, self.risk_params['target_vol'])

    def compute_weights(self, 
                        regime: str, 
                        vol_forecast: float, 
                        historical_returns: pd.DataFrame,
                        asset_names: List[str]) -> Dict[str, float]:
        """
        Step 4.3: Generate final capital weights using Dynamic Volatility Targeting.
        
        Formula: Weight = Target_Volatility / Forecasted_Volatility
        Constrained by: max_leverage
        """
        target_vol = self.get_regime_target_vol(regime)
        
        # 1. Simple Volatility Targeting for the primary asset (NIFTY50)
        # Weight = Target_Vol / Forecasted_Vol
        # e.g., 0.15 (Target) / 0.20 (Forecast) = 0.75 weight
        raw_weight = target_vol / vol_forecast
        
        # 2. Apply Leverage Capping
        final_equity_weight = min(raw_weight, self.max_leverage)
        
        # 3. Defensive Allocation (Diversification into Gold/Cash)
        # In this simplified Trinity 1.0, we allocate the 'Leftover' capital 
        # based on the regime. 
        # Bull: Equity Focus
        # Neutral: Balanced (Nifty + Gold)
        # Bear: Protective (Gold + Cash/USDINR)
        
        weights = {}
        equity_name = asset_names[0] # Assumes NIFTY50
        commodity_name = asset_names[1] # Assumes Gold
        currency_name = asset_names[2] # Assumes USDINR
        
        if regime == 'Bull':
            weights[equity_name] = final_equity_weight
            weights[commodity_name] = max(0, (self.max_leverage - final_equity_weight) * 0.3)
            weights[currency_name] = 0.0
        elif regime == 'Neutral':
            weights[equity_name] = final_equity_weight * 0.7
            weights[commodity_name] = (self.max_leverage - weights[equity_name]) * 0.5
            weights[currency_name] = max(0, self.max_leverage - weights[equity_name] - weights[commodity_name])
        else: # Bear
            weights[equity_name] = final_equity_weight * 0.2
            weights[commodity_name] = (self.max_leverage - weights[equity_name]) * 0.6
            weights[currency_name] = max(0, self.max_leverage - weights[equity_name] - weights[commodity_name])
            
        # Add Cash component
        total_allocated = sum(weights.values())
        weights['Cash'] = max(0, 1.0 - total_allocated)
        
        return weights

    def generate_strategy_timeline(self, 
                                   df: pd.DataFrame, 
                                   vol_col: str, 
                                   regime_col: str,
                                   asset_names: List[str]) -> pd.DataFrame:
        """
        Generate a historical timeline of target weights.
        """
        strategy_data = []
        
        for idx, row in df.iterrows():
            w = self.compute_weights(
                regime=row[regime_col],
                vol_forecast=row[vol_col],
                historical_returns=None, # Not used in simple vol-targeting yet
                asset_names=asset_names
            )
            w['Date'] = idx
            strategy_data.append(w)
            
        return pd.DataFrame(strategy_data).set_index('Date')
