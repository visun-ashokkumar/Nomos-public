import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib
import os
from typing import List, Optional

class NomosHMM:
    """
    Hidden Markov Model wrapper for Project Nomos market regime detection.
    Uses Gaussian Emissions.
    """
    
    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            random_state=random_state,
            n_iter=1000
        )
        self.feature_columns = []
        self.state_labels = {}

    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Fit the HMM to the provided features.
        """
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        
        # Fit the model
        self.model.fit(X)
        
        # Check for convergence
        if not self.model.monitor_.converged:
            print("Warning: HMM did not converge! Try more iterations or more data.")
        
        # Automatically label states based on Mean Returns of the primary asset (first feature)
        self._map_states(X)

    def _map_states(self, X: np.ndarray):
        """
        Heuristic to map internal state IDs to logical labels (Bull, Bear, Neutral).
        Assumes first column is the primary Return series (e.g. NIFTY50_Ret).
        """
        means = self.model.means_[:, 0]  # First feature mean
        sorted_indices = np.argsort(means)  # Sort by returns (Ascending)
        
        # Map: 
        # Lowest Mean -> "Crisis/Bear"
        # Middle Mean -> "Neutral/Side"
        # Highest Mean -> "Bull"
        if self.n_components == 3:
            self.state_labels = {
                sorted_indices[0]: "Bear",
                sorted_indices[1]: "Neutral",
                sorted_indices[2]: "Bull"
            }
        else:
            # Generic labeling if not 3 states
            for i, idx in enumerate(sorted_indices):
                self.state_labels[idx] = f"Regime_{i}"

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict the hidden states for the given features.
        Returns a series of labels.
        """
        X = df[self.feature_columns].values
        states = self.model.predict(X)
        return pd.Series([self.state_labels.get(s, f"State_{s}") for s in states], index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the probability of each state.
        """
        X = df[self.feature_columns].values
        probas = self.model.predict_proba(X)
        # Sort column names according to state_labels
        col_names = [self.state_labels.get(i, f"State_{i}") for i in range(self.n_components)]
        return pd.DataFrame(probas, index=df.index, columns=col_names)

    def save_model(self, path: str):
        """
        Persist model to disk.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path: str) -> 'NomosHMM':
        """
        Load model from disk.
        """
        return joblib.load(path)
