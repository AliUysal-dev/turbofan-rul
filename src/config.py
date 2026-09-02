"""
src/config.py
Uygulama konfigürasyonu ve hiperparametre yönetimi.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass(frozen=True)
class PathConfig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"


@dataclass(frozen=True)
class FeatureConfig:
    MAX_RUL: int = 125
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [5, 20])
    CONSTANT_FEATURES: List[str] = field(default_factory=lambda: [
        "sensor_1", "sensor_5", "sensor_6", "sensor_10",
        "sensor_16", "sensor_18", "sensor_19", "setting_3"
    ])


@dataclass(frozen=True)
class ModelConfig:
    EXPERIMENT_NAME: str = "turbofan-rul-production"
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    XGB_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    })