import numpy as np
import pandas as pd
import pytest

from src.features import generate_time_series_features


def test_k03_interleaved_engines_delta_alignment():
    """
    K-03 Regresyon Testi:
    Farklı motorlara ait satırlar sıralı olmadan iç içe (interleaved) verildiğinde,
    delta_20 ve kayan ortalamaların diğer motorun değerleriyle karışmadığını doğrular.
    """
    # 1. Bilerek iç içe geçmiş sentetik veri seti:
    # Motor 1: 10, 20 değerleri (Ortalama: 15, Delta: +5)
    # Motor 2: 1000, 2000 değerleri (Ortalama: 1500, Delta: +500)
    raw_data = {
        "unit_number": [1, 2, 1, 2],
        "time_in_cycles": [1, 1, 2, 2],
        "setting_1": [0.0, 0.0, 0.0, 0.0],
        "setting_2": [0.0, 0.0, 0.0, 0.0],
        "sensor_2": [10.0, 1000.0, 20.0, 2000.0]
    }
    df_interleaved = pd.DataFrame(raw_data)

    # 2. Özellik üretimi fonksiyonunu çalıştır
    df_result = generate_time_series_features(
        df_interleaved,
        sensors=["sensor_2"],
        settings=["setting_1", "setting_2"]
    )

    # 3. Motor 1'in 2. döngüsünü filtrele
    m1_c2 = df_result[(df_result["unit_number"] == 1) & (df_result["time_in_cycles"] == 2)].iloc[0]
    # Beklenen: rolling_mean_20 = (10 + 20) / 2 = 15.0 | delta_20 = 20.0 - 15.0 = 5.0
    assert np.isclose(m1_c2["sensor_2_rolling_mean_20"], 15.0), "Motor 1 kayan ortalaması hatalı hesaplandı!"
    assert np.isclose(m1_c2["sensor_2_delta_20"], 5.0), "K-03 Hatası: Motor 1 deltası başka motorla karıştı!"

    # 4. Motor 2'nin 2. döngüsünü filtrele
    m2_c2 = df_result[(df_result["unit_number"] == 2) & (df_result["time_in_cycles"] == 2)].iloc[0]
    # Beklenen: rolling_mean_20 = (1000 + 2000) / 2 = 1500.0 | delta_20 = 2000.0 - 1500.0 = 500.0
    assert np.isclose(m2_c2["sensor_2_rolling_mean_20"], 1500.0), "Motor 2 kayan ortalaması hatalı hesaplandı!"
    assert np.isclose(m2_c2["sensor_2_delta_20"], 500.0), "K-03 Hatası: Motor 2 deltası başka motorla karıştı!"


def test_feature_count_is_81():
    """
    Ö-01 Doğrulama Testi:
    14 sensör + 2 ayar kolonu üzerinden üretilen toplam özellik sayısının
    tam olarak 81 kolon olduğunu teyit eder.
    """
    # 20 döngülük tek motorluk kukla veri
    cycles = 20
    data = {
        "unit_number": [1] * cycles,
        "time_in_cycles": list(range(1, cycles + 1)),
        "setting_1": [0.0] * cycles,
        "setting_2": [0.0] * cycles,
    }
    # 14 zorunlu sensörü ekle
    for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
        data[f"sensor_{s}"] = [100.0] * cycles

    df_sample = pd.DataFrame(data)
    df_features = generate_time_series_features(df_sample)

    # 16 değişken x 5 türetim (ham, rm5, std5, rm20, d20) + time_in_cycles = 81
    engineered_cols = [c for c in df_features.columns if c != "unit_number"]
    assert len(engineered_cols) == 81, f"Özellik sayısı 81 olmalı, bulunan: {len(engineered_cols)}"