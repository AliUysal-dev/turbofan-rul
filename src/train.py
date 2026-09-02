"""
src/train.py
Uçtan uca model eğitim, değerlendirme ve MLflow kayıt boru hattı.
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
import mlflow.xgboost

from config import PathConfig, FeatureConfig, ModelConfig
from features import (
    load_raw_data,
    add_piecewise_rul,
    generate_time_series_features,
    get_feature_columns,
    INFORMATIVE_SENSORS,
)
from metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class TurbofanTrainingPipeline:
    """Model eğitim döngüsünü ve artifact kayıtlarını yöneten boru hattı."""

    def __init__(self):
        self.paths = PathConfig()
        self.feature_cfg = FeatureConfig()
        self.model_cfg = ModelConfig()

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Ham eğitim, test ve yer gerçeği (ground truth) RUL verilerini yükler."""
        logger.info("Veri kümeleri diskten okunuyor: %s", self.paths.DATA_RAW_DIR)
        train_df = load_raw_data(self.paths.DATA_RAW_DIR / "train_FD001.txt")
        test_df = load_raw_data(self.paths.DATA_RAW_DIR / "test_FD001.txt")
        
        y_test_raw = pd.read_csv(
            self.paths.DATA_RAW_DIR / "RUL_FD001.txt", 
            sep=r"\s+", 
            header=None, 
            names=["true_rul"]
        )
        y_test = y_test_raw["true_rul"].clip(upper=self.feature_cfg.MAX_RUL).values
        return train_df, test_df, y_test

    def run(self) -> None:
        """Tüm eğitim ve kayıt sürecini yürütür."""
        train_df, test_df, y_test = self.load_datasets()

        logger.info("Zaman serisi özellik dönüşümleri uygulanıyor.")
        train_df = add_piecewise_rul(train_df, max_rul=self.feature_cfg.MAX_RUL)
        train_feat = generate_time_series_features(train_df, INFORMATIVE_SENSORS)
        test_feat = generate_time_series_features(test_df, INFORMATIVE_SENSORS)

        feature_cols = get_feature_columns(train_feat)
        logger.info("Aktif özellik sayısı: %d", len(feature_cols))

        # Motor bazlı grup ayrımı
        unique_units = train_feat["unit_number"].unique()
        train_units, val_units = train_test_split(
            unique_units, 
            test_size=self.model_cfg.TEST_SIZE, 
            random_state=self.model_cfg.RANDOM_STATE
        )

        X_train = train_feat[train_feat["unit_number"].isin(train_units)][feature_cols]
        y_train = train_feat[train_feat["unit_number"].isin(train_units)]["RUL_clipped"]

        X_val = train_feat[train_feat["unit_number"].isin(val_units)][feature_cols]
        y_val = train_feat[train_feat["unit_number"].isin(val_units)]["RUL_clipped"]

        # Test seti son döngü filtrelemesi
        test_last = test_feat.groupby("unit_number").last().reset_index()
        X_test = test_last[feature_cols]

        logger.info("XGBoost modeli optimize ediliyor...")
        model = xgb.XGBRegressor(**self.model_cfg.XGB_PARAMS)
        model.fit(X_train, y_train)

        # Doğrulama ve Test metrikleri
        val_metrics = calculate_metrics(y_val, model.predict(X_val))
        test_metrics = calculate_metrics(y_test, model.predict(X_test))

        logger.info(
            "Validation Skorları -> RMSE: %.2f | MAE: %.2f | PHM: %s",
            val_metrics["rmse"], val_metrics["mae"], f"{val_metrics['phm_score']:,.2f}"
        )
        logger.info(
            "NASA Test Skorları  -> RMSE: %.2f | MAE: %.2f | PHM: %s",
            test_metrics["rmse"], test_metrics["mae"], f"{test_metrics['phm_score']:,.2f}"
        )

        self._log_to_mlflow(model, feature_cols, val_metrics, test_metrics)

    def _log_to_mlflow(self, model: xgb.XGBRegressor, feature_cols: list, val_metrics: dict, test_metrics: dict) -> None:
        """Deney sonuçlarını ve model artifact'lerini MLflow'a kaydeder."""
        mlflow.set_tracking_uri(f"file:{self.paths.MLRUNS_DIR}")
        mlflow.set_experiment(self.model_cfg.EXPERIMENT_NAME)

        with mlflow.start_run(run_name="production_candidate_xgboost"):
            mlflow.log_params(self.model_cfg.XGB_PARAMS)
            mlflow.log_param("max_rul_clip", self.feature_cfg.MAX_RUL)
            mlflow.log_param("num_features", len(feature_cols))

            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.xgboost.log_model(model, name="model")
            logger.info("Model ve metrikler MLflow Registry/Artifact Store'a başarıyla mühürlendi.")


if __name__ == "__main__":
    pipeline = TurbofanTrainingPipeline()
    pipeline.run()