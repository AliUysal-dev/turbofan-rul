from typing import List, Optional
import pandas as pd

# Modelin kullandığı 14 bilgilendirici sensör
INFORMATIVE_SENSORS: List[str] = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

# Özellik türetimine dahil edilen 2 operasyonel ayar
INFORMATIVE_SETTINGS: List[str] = ["setting_1", "setting_2"]


def generate_time_series_features(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None,
    settings: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Zaman serisi telemetri verilerinden kayan pencere (rolling) ve 
    fark (delta) özelliklerini türetir.
    
    K-03 Çözümü:
    1. df.sort_values(["unit_number", "time_in_cycles"]) ile sıralama garantiye alınır.
    2. Kayan istatistikler motor bazında (unit_number) izole hesaplanır.
    3. Delta hesabı .values (pozisyonel) yerine indeks eşleşmesiyle yapılır.
       Böylece karışık sıralı veya iç içe geçmiş motor verilerinde kayma oluşmaz.
    """
    if df.empty:
        return df.copy()

    # 1. Sıralama güvencesi (K-03 çözümü)
    df_sorted = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    target_sensors = sensors if sensors is not None else INFORMATIVE_SENSORS
    target_settings = settings if settings is not None else INFORMATIVE_SETTINGS
    
    # 14 sensör + 2 ayar = 16 dönüşüm kolonu
    transform_cols = [c for c in (target_sensors + target_settings) if c in df_sorted.columns]

    # 2. Motor bazında kayan pencere istatistikleri
    grouped = df_sorted.groupby("unit_number")[transform_cols]

    roll_mean_5 = grouped.rolling(window=5, min_periods=1).mean().droplevel(0)
    roll_std_5 = grouped.rolling(window=5, min_periods=1).std().droplevel(0).fillna(0.0)
    roll_mean_20 = grouped.rolling(window=20, min_periods=1).mean().droplevel(0)

    # 3. İndeks hizalı delta hesabı (Kesinlikle .values kullanılmaz)
    deltas_20 = df_sorted[transform_cols].sub(roll_mean_20)

    # 4. Sütun adlandırmaları
    roll_mean_5.columns = [f"{c}_rolling_mean_5" for c in transform_cols]
    roll_std_5.columns = [f"{c}_rolling_std_5" for c in transform_cols]
    roll_mean_20_renamed = roll_mean_20.copy()
    roll_mean_20_renamed.columns = [f"{c}_rolling_mean_20" for c in transform_cols]
    deltas_20.columns = [f"{c}_delta_20" for c in transform_cols]

    # 5. Özellikleri birleştir (16 ham + 64 türetilmiş + time_in_cycles = 81 özellik)
    df_features = pd.concat(
        [df_sorted, roll_mean_5, roll_std_5, roll_mean_20_renamed, deltas_20],
        axis=1
    )

    return df_features