from kiteconnect import KiteConnect
import pandas as pd
from src.data.ingestor import DataIngestor
from typing import List, Optional
import os

class KiteIngestor(DataIngestor):
    """
    Ingestor implementation for Kite Connect (Zerodha).
    Specifically used for India VIX.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = self.config['credentials']['kite']['api_key']
        self.api_secret = self.config['credentials']['kite']['api_secret']
        self.access_token = self.config['credentials']['kite'].get('access_token')
        
        self.kite = KiteConnect(api_key=self.api_key)
        if self.access_token:
            self.kite.set_access_token(self.access_token)

    def fetch_data(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data from Kite Connect using a chunking strategy
        to bypass the 2000-candle limit (approx 5.5 years for 'day' interval).
        """
        if not self.access_token:
            raise ValueError("Kite Access Token is missing. Please authenticate and set the access_token in config or environment.")

        vix_config = self.config['assets']['volatility']['vix']
        instrument_token = vix_config['instrument_token']
        interval = self.config['parameters']['data']['frequency']
        
        # Define the overall date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
        print(f"Fetching {vix_config['name']} from Kite: {start_dt.date()} to {end_dt.date()} in chunks...")

        all_records = []
        current_start = start_dt
        
        # Max chunk for 'day' is 2000 days per documentation/testing
        step_days = 2000 
        
        while current_start < end_dt:
            current_end = min(current_start + pd.Timedelta(days=step_days), end_dt)
            
            try:
                # Format dates as YYYY-MM-DD HH:MM:SS as per docs
                from_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
                to_str = current_end.strftime('%Y-%m-%d %H:%M:%S')
                
                chunk = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_str,
                    to_date=to_str,
                    interval=interval
                )
                
                if chunk:
                    all_records.extend(chunk)
                
                # Move to next window (next start = current end + 1 min to avoid overlap)
                current_start = current_end + pd.Timedelta(minutes=1)
                
            except Exception as e:
                print(f"Error fetching chunk {from_str} to {to_str}: {e}")
                break
                
        if not all_records:
            return pd.DataFrame()
            
        # 1. Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # 2. Process Index
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df.set_index('date', inplace=True)
        
        # 3. Return only the 'close' price
        vix_df = df[['close']].copy()
        vix_df.columns = [vix_config['name']]
        
        # 4. Remove any duplicate dates
        vix_df = vix_df[~vix_df.index.duplicated(keep='first')]
        
        return vix_df
