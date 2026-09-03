import numpy as np
import pytest

from src.metrics import compute_phm_score


def test_phm_score_known_values():
    """
    Ö-07: NASA PHM 2008 asimetrik skor fonksiyonunun
    erken (d < 0) ve geç (d >= 0) tahminlerdeki teorik davranışını denetler.
    """
    # 1. d = 0 (Kusursuz tahmin) -> Ceza 0 olmalı
    y_true_exact = np.array([100.0])
    y_pred_exact = np.array([100.0])
    assert np.isclose(compute_phm_score(y_true_exact, y_pred_exact), 0.0)

    # 2. Erken tahmin: d = -13 -> exp(-(-13)/13) - 1 = e^1 - 1 ≈ 1.718
    y_true_early = np.array([100.0])
    y_pred_early = np.array([87.0])
    expected_early = np.exp(1) - 1
    assert np.isclose(compute_phm_score(y_true_early, y_pred_early), expected_early, atol=1e-2)

    # 3. Geç tahmin: d = +10 -> exp(10/10) - 1 = e^1 - 1 ≈ 1.718
    y_true_late = np.array([100.0])
    y_pred_late = np.array([110.0])
    expected_late = np.exp(1) - 1
    assert np.isclose(compute_phm_score(y_true_late, y_pred_late), expected_late, atol=1e-2)