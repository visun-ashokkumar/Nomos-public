import yaml
import pandas as pd
import os
from typing import Optional
from src.data.yahoo_ingestor import YahooIngestor
from src.data.kite_ingestor import KiteIngestor
from src.data.processor import DataProcessor

class DataManager:
    """
    Orchestrates data ingestion from multiple sources and manages data persistence.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.yahoo_ingestor = YahooIngestor(self.config)
        self.kite_ingestor = KiteIngestor(self.config)
        self.processor = DataProcessor(self.config)

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrate full feature engineering for Step 1.3 & 1.4.
        """
        df_processed = df_raw.copy()
        
        # 1. Log-Returns (1.3)
        trinity_names = [
            self.config['assets']['trinity']['equity']['name'],
            self.config['assets']['trinity']['commodity']['name'],
            self.config['assets']['trinity']['currency']['name']
        ]
        df_processed = self.processor.compute_log_returns(df_processed, trinity_names)
        
        # 2. VIX Spread (1.4)
        vix_name = self.config['assets']['volatility']['vix']['name']
        vix_window = self.config['parameters']['data'].get('lookback_vix_ma', 20)
        df_processed = self.processor.compute_vix_spread(df_processed, vix_name, window=vix_window)
        
        # 3. Detrended Volume (1.4) - Focused on NIFTY50
        equity_name = self.config['assets']['trinity']['equity']['name']
        vol_col = f"{equity_name}_Volume"
        vol_window = self.config['parameters']['data'].get('lookback_volume_avg', 10)
        if vol_col in df_processed.columns:
            df_processed = self.processor.compute_detrended_volume(df_processed, vol_col, window=vol_window)
            
        # 4. Rolling Correlations (1.4)
        # Corr(Nifty_Ret, Gold_Ret) and Corr(Nifty_Ret, USDINR_Ret)
        nifty_ret = f"{equity_name}_Ret"
        target_rets = [f"{self.config['assets']['trinity']['commodity']['name']}_Ret", 
                       f"{self.config['assets']['trinity']['currency']['name']}_Ret"]
        df_processed = self.processor.compute_rolling_correlations(df_processed, nifty_ret, target_rets, window=20)
        
        # Drop NaNs created by rolling windows
        df_processed.dropna(inplace=True)
        
        return df_processed
    
    def fetch_all_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch Trinity assets from Yahoo and VIX from Kite, then merge.
        Aligns Price and Volume for all assets.
        """
        # 0. Set date range
        start_date = start_date if start_date else self.config['parameters']['data']['start_date']
        end_date = end_date if end_date else self.config['parameters']['data'].get('end_date')
        
        # 1. Fetch Trinity Assets (NIFTY, Gold, USDINR)
        trinity_tickers = [
            self.config['assets']['trinity']['equity']['ticker'],
            self.config['assets']['trinity']['commodity']['ticker'],
            self.config['assets']['trinity']['currency']['ticker']
        ]
        
        df_yahoo = self.yahoo_ingestor.fetch_data(
            trinity_tickers, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # Normalize Yahoo Index to Date only
        df_yahoo.index = pd.to_datetime(df_yahoo.index).date
        
        # 1a. Rename/Flatten Yahoo columns: (Price, Ticker) -> 'NIFTY50', (Volume, Ticker) -> 'NIFTY50_Volume'
        ticker_map = {
            self.config['assets']['trinity']['equity']['ticker']: self.config['assets']['trinity']['equity']['name'],
            self.config['assets']['trinity']['commodity']['ticker']: self.config['assets']['trinity']['commodity']['name'],
            self.config['assets']['trinity']['currency']['ticker']: self.config['assets']['trinity']['currency']['name']
        }
        
        new_cols = []
        for type_label, ticker in df_yahoo.columns:
            friendly = ticker_map.get(ticker, ticker)
            if type_label == 'Price':
                new_cols.append(friendly)
            else:
                new_cols.append(f"{friendly}_{type_label}")
        df_yahoo.columns = new_cols
        
        # 2. Fetch India VIX from Kite
        try:
            df_vix = self.kite_ingestor.fetch_data(
                [self.config['assets']['volatility']['vix']['ticker']],
                start_date=start_date,
                end_date=end_date
            )
            if not df_vix.empty:
                # Normalize Kite Index to Date only
                df_vix.index = pd.to_datetime(df_vix.index).date
                # Specifically for VIX, we keep OHLC but our processor might only need Close/Volume for now
                # We'll prefix all VIX columns
                vix_name = self.config['assets']['volatility']['vix']['name']
                # For simplicity in this project, we only grab the 'Close' as the primary VIX level
                df_vix = df_vix[['Close']].copy()
                df_vix.columns = [vix_name]
        except Exception as e:
            print(f"Skipping Kite ingestion due to error: {e}")
            df_vix = pd.DataFrame()

        # 3. Merge DataFrames
        if not df_vix.empty:
            df_combined = df_yahoo.join(df_vix, how='inner')
        else:
            df_combined = df_yahoo
            
        # 4. Final Cleaning
        df_combined.index = pd.to_datetime(df_combined.index)
        df_combined.sort_index(inplace=True)
        df_combined.ffill(inplace=True)
        # We don't dropna yet, process_data will handle it after features are built
        
        return df_combined

    def save_raw_data(self, df: pd.DataFrame, filename: str = "raw_trinity.csv"):
        """
        Persist raw data to the data directory.
        """
        raw_dir = self.config['paths']['data_raw']
        os.makedirs(raw_dir, exist_ok=True)
        
        path = os.path.join(raw_dir, filename)
        df.to_csv(path)
        print(f"Saved raw data to {path}")

if __name__ == "__main__":
    # Test execution
    manager = DataManager()
    try:
        combined_data = manager.fetch_all_data()
        print("Successfully combined data:")
        print(combined_data.head())
        manager.save_raw_data(combined_data)
    except Exception as e:
        print(f"Ingestion Failed: {e}")
