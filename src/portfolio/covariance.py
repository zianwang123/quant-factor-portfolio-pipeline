import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Standard sample covariance matrix."""
    return returns.cov()


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage estimator for covariance.

    Shrinks toward a structured target (scaled identity) to reduce estimation error.
    Especially important when N (assets) is large relative to T (observations).
    """
    clean = returns.dropna()
    lw = LedoitWolf().fit(clean)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def exponential_weighted_covariance(returns: pd.DataFrame, halflife: int = 36) -> pd.DataFrame:
    """Exponentially weighted covariance with specified halflife (in months)."""
    ewm = returns.ewm(halflife=halflife).cov()
    # Get the last date's covariance
    last_date = returns.index[-1]
    cov_matrix = ewm.loc[last_date]
    return cov_matrix


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Find the nearest positive semi-definite matrix using eigenvalue clipping."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
