import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class NomosBacktester:
    """
    Backtesting & Performance Analytics Engine for Project Nomos.
    Simulates strategy execution including transaction costs and slippage.
    """

    def __init__(self, config: dict):
        self.config = config
        self.transaction_cost = config['parameters'].get('transaction_cost', 0.001) # Default 10 bps

    def run_backtest(self, 
                     returns_df: pd.DataFrame, 
                     weights_df: pd.DataFrame, 
                     benchmark_col: str = 'NIFTY50_Ret') -> pd.DataFrame:
        """
        Step 5.1 & 5.2: Calculate the strategy equity curve and friction costs.
        
        Logic: 
        Strategy_Ret[t] = Sum(Weight[t-1] * Asset_Ret[t]) - Transaction_Costs
        """
        # Ensure indices are aligned
        common_idx = returns_df.index.intersection(weights_df.index)
        rets = returns_df.loc[common_idx]
        weights = weights_df.loc[common_idx]
        
        # 1. Shift weights by 1 day (Allocation decided at T-1 affects returns at T)
        target_weights = weights.shift(1)
        
        # 2. Portfolio Returns (Before costs)
        # We find the corresponding '_Ret' column for each asset weight
        asset_cols = [c for c in weights.columns if c != 'Cash']
        portfolio_rets = pd.Series(0, index=common_idx)
        
        for asset in asset_cols:
            ret_col = f"{asset}_Ret"
            if ret_col in rets.columns:
                portfolio_rets += target_weights[asset] * rets[ret_col]
            else:
                print(f"WARNING: Return column {ret_col} not found for asset {asset}. Skipping.")
        
        # 3. Transaction Costs
        # Turnover = Sum(|Weight_t - Weight_{t-1}|)
        # Cost = Turnover * Transaction_Cost_Rate
        weight_diff = weights[asset_cols].diff().abs().sum(axis=1)
        costs = weight_diff * self.transaction_cost
        
        # Net Returns
        net_rets = portfolio_rets - costs.shift(1).fillna(0)
        
        # 4. Construct Benchmark
        benchmark_rets = rets[benchmark_col]
        
        # 5. Build Result DataFrame
        results = pd.DataFrame({
            'Strategy_Ret': net_rets,
            'Benchmark_Ret': benchmark_rets,
            'Turnover': weight_diff,
            'Costs': costs
        }, index=common_idx)
        
        # 6. Cumulative Curves (Wealth indices starting at 1.0)
        results['Strategy_Equity'] = (1 + results['Strategy_Ret']).cumprod()
        results['Benchmark_Equity'] = (1 + results['Benchmark_Ret']).cumprod()
        
        return results

    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate key performance indicators (KPIs).
        """
        def get_stats(rets):
            ann_ret = rets.mean() * 252
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
            
            # Drawdown
            equity = (1 + rets).cumprod()
            peak = equity.cummax()
            dd = (equity - peak) / peak
            max_dd = dd.min()
            
            return {
                'CAGR': ann_ret,
                'Volatility': ann_vol,
                'Sharpe': sharpe,
                'MaxDD': max_dd
            }
            
        strat_metrics = get_stats(results['Strategy_Ret'])
        bench_metrics = get_stats(results['Benchmark_Ret'])
        
        return {
            'Strategy': strat_metrics,
            'Benchmark': bench_metrics
        }
