"""
src/features.py
Veri okuma, ön işleme ve zaman serisi özellik mühendisliği fonksiyonları.
"""
import pandas as pd

# Sütun Tanımları
INDEX_COLS = ["unit_number", "time_in_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

CONSTANT_FEATURES = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19", "setting_3"
]

INFORMATIVE_SENSORS = [
    col for col in (SETTING_COLS + SENSOR_COLS)
    if col not in CONSTANT_FEATURES
]

MAX_RUL = 125


def load_raw_data(filepath: str) -> pd.DataFrame:
    """C-MAPSS boşlukla ayrılmış ham veri setini yükler."""
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=ALL_COLUMNS)


def add_piecewise_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """Eğitim verisine gerçek ve kırpılmış (piecewise) RUL etiketlerini ekler."""
    df_out = df.copy()
    max_cycle = df_out.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    df_out = df_out.merge(max_cycle, on="unit_number", how="left")
    df_out["RUL"] = df_out["max_cycle"] - df_out["time_in_cycles"]
    df_out["RUL_clipped"] = df_out["RUL"].clip(upper=max_rul)
    return df_out


def generate_time_series_features(df: pd.DataFrame, sensors: list = INFORMATIVE_SENSORS) -> pd.DataFrame:
    """
    Motor bazında 5 ve 20 döngülük kayan ortalama/sapma ve 
    20 döngülük delta özelliklerini türetir.
    """
    df_feat = df.copy()

    # 1. 5 döngülük pencereler (mean & std)
    roll_mean_5 = (
        df_feat.groupby("unit_number")[sensors]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_mean_5.columns = [f"{col}_roll_mean_5" for col in sensors]

    roll_std_5 = (
        df_feat.groupby("unit_number")[sensors]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    roll_std_5.columns = [f"{col}_roll_std_5" for col in sensors]

    # 2. 20 döngülük kayan ortalama
    roll_mean_20 = (
        df_feat.groupby("unit_number")[sensors]
        .rolling(window=20, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_mean_20.columns = [f"{col}_roll_mean_20" for col in sensors]

    # 3. Delta Özellikleri (Anlık - 20 Trendi)
    deltas = pd.DataFrame(
        df_feat[sensors].values - roll_mean_20.values,
        columns=[f"{col}_delta_20" for col in sensors],
        index=df_feat.index
    )

    return pd.concat([df_feat, roll_mean_5, roll_std_5, roll_mean_20, deltas], axis=1)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Eğitimde girdi olarak kullanılacak özellik sütunlarını filtreler."""
    excluded = CONSTANT_FEATURES + ["unit_number", "max_cycle", "RUL", "RUL_clipped"]
    return [c for c in df.columns if c not in excluded]