"""
src/metrics.py
Model değerlendirme ve NASA PHM asimetrik hata metrikleri.
"""
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def compute_phm_score(y_true, y_pred) -> float:
    """
    NASA PHM 2008 Challenge asimetrik ceza skoru.
    d = y_pred - y_true
    d < 0  (erken tahmin): exp(-d / 13) - 1
    d >= 0 (geç tahmin):   exp(d / 10) - 1
    """
    d = np.array(y_pred) - np.array(y_true)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def calculate_metrics(y_true, y_pred) -> dict:
    """Temel regresyon metriklerini ve PHM skorunu sözlük olarak döner."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "phm_score": compute_phm_score(y_true, y_pred),
    }