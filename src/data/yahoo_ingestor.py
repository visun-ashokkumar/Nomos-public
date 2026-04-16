import yfinance as yf
import pandas as pd
from src.data.ingestor import DataIngestor
from typing import List, Optional

class YahooIngestor(DataIngestor):
    """
    Ingestor implementation for Yahoo Finance.
    """
    
    def fetch_data(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        """
        print(f"Fetching data from Yahoo Finance for: {tickers}")
        
        # yfinance download returns a multi-index DataFrame for multiple tickers
        # We fetch Open, High, Low, Close, Volume
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for tickers: {tickers}")
            
        # Select best available close price and Volume
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        
        if price_col in data.columns and 'Volume' in data.columns:
            prices = data[price_col]
            volumes = data['Volume']
            
            # Combine into a single DF with MultiIndex for consistency
            out_df = pd.concat([prices, volumes], axis=1, keys=['Price', 'Volume'])
        else:
            raise KeyError(f"Could not find required columns in data. Columns: {data.columns}")

        # Ensure index is datetime
        out_df.index = pd.to_datetime(out_df.index)
        
        return out_df
