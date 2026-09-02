"""
src/api/app.py
Kestirimci Bakım RUL Çıkarım Servisi (FastAPI).
"""
import os
import sys
from pathlib import Path

# src dizinini Python modül arama yolunun en başına ekler
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from contextlib import asynccontextmanager
import pandas as pd
import mlflow
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, status

from config import PathConfig, ModelConfig, FeatureConfig
from features import (
    generate_time_series_features, 
    get_feature_columns, 
    INFORMATIVE_SENSORS,
    ALL_COLUMNS
)
from api.schemas import PredictionRequest, PredictionResponse
from utils.logger import get_logger

logger = get_logger("turbofan_api")

# Model ve çalışma durumunu tutan servis önbelleği
model_cache = {
    "model": None,
    "run_id": None,
    "feature_cols": []
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama ayağa kalkarken MLflow'dan en son eğitilen modeli yükler."""
    paths = PathConfig()
    model_cfg = ModelConfig()
    
    mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")
    experiment = mlflow.get_experiment_by_name(model_cfg.EXPERIMENT_NAME)
    
    if not experiment:
        logger.error("MLflow deneyi bulunamadı: %s", model_cfg.EXPERIMENT_NAME)
        raise RuntimeError(f"MLflow deneyi bulunamadı: {model_cfg.EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        logger.error("Deney içinde kayıtlı aktif model koşusu (run) bulunamadı.")
        raise RuntimeError("Model koşusu bulunamadı.")

    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"
    
    logger.info("Üretim modeli MLflow'dan yükleniyor: %s", model_uri)
    model_cache["model"] = mlflow.xgboost.load_model(model_uri)
    model_cache["run_id"] = latest_run_id
    logger.info("Model başarıyla belleğe alındı. Servis istek kabul etmeye hazır.")

    yield
    model_cache.clear()


app = FastAPI(
    title="C-MAPSS Turbofan Engine RUL API",
    description="Uçak motorları telemetrisi üzerinden Kalan Faydalı Ömür (RUL) kestirimi sağlayan MLOps servisi.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Servis ve model yüklenme durumunu kontrol eder."""
    return {
        "status": "online",
        "model_loaded": model_cache["model"] is not None,
        "active_run_id": model_cache["run_id"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_rul(payload: PredictionRequest):
    """Gelen telemetri geçmişinden dinamik özellikler türetip motor için RUL tahmini üretir."""
    if model_cache["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model henüz belleğe yüklenmedi."
        )

    # 1. İstek gövdesini (JSON) DataFrame formatına dönüştür
    records = []
    for cycle in payload.history:
        row = {
            "unit_number": payload.unit_number,
            "time_in_cycles": cycle.time_in_cycles,
            "setting_1": cycle.setting_1,
            "setting_2": cycle.setting_2,
            "setting_3": cycle.setting_3,
        }
        row.update(cycle.sensors)
        records.append(row)

    df_incoming = pd.DataFrame(records)
    
    # Eksik sensör sütunlarını sıfır ile doldur (sabit sensörler dahil)
    for col in ALL_COLUMNS:
        if col not in df_incoming.columns:
            df_incoming[col] = 0.0

    # Kronolojik döngü sırasına diz
    df_incoming = df_incoming.sort_values("time_in_cycles").reset_index(drop=True)

    # 2. Özellik mühendisliği (5/20 kayan ortalama, sapma ve delta)
    df_feat = generate_time_series_features(df_incoming, INFORMATIVE_SENSORS)
    feature_cols = get_feature_columns(df_feat)

    # Yalnızca en son döngüyü çıkarım için seç
    latest_state = df_feat.iloc[[-1]][feature_cols]
    current_cycle = int(df_incoming["time_in_cycles"].iloc[-1])

    # 3. Model çıkarımı
    predicted_rul = float(model_cache["model"].predict(latest_state)[0])
    predicted_rul = max(0.0, predicted_rul)

    # 4. Operasyonel karar kuralı
    if predicted_rul <= 25.0:
        health_status = "CRITICAL"
        action = "Acil bakım planla; motor bir sonraki uçuştan önce hangara çekilmeli."
    elif predicted_rul <= 50.0:
        health_status = "WARNING"
        action = "Gözlem sıklığını artır; planlı bakım penceresi rezerve et."
    else:
        health_status = "HEALTHY"
        action = "Motor normal operasyon limitlerinde çalışıyor."

    return PredictionResponse(
        unit_number=payload.unit_number,
        current_cycle=current_cycle,
        predicted_rul=round(predicted_rul, 2),
        health_status=health_status,
        recommended_action=action
    )