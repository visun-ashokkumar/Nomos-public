from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class DataIngestor(ABC):
    """
    Abstract Base Class for data ingestion from various providers.
    """
    
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fetch_data(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data for a list of tickers.
        """
        pass
