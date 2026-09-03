from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

from src.utils.logger import get_logger
from src.api.schemas import PredictionRequest, PredictionResponse
from src.features import generate_time_series_features, INFORMATIVE_SENSORS

# Ö-11: Merkezi loglama entegrasyonu[cite: 1, 2]
logger = get_logger("turbofan_api")

# Global model önbelleği
model_cache: Dict[str, Any] = {}

# K-01 & Ö-06: Sabit üretim modeli yolu[cite: 1, 2]
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "production_model"
FALLBACK_MODEL_DIR = Path("models/production_model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI yaşam döngüsü: Sabit şampiyon XGBoost modelini belleğe alır[cite: 1, 2]."""
    logger.info("Turbofan RUL API başlatılıyor...")

    target_path = MODEL_DIR if MODEL_DIR.exists() else FALLBACK_MODEL_DIR

    if not (target_path / "MLmodel").exists():
        logger.critical("Üretim modeli bulunamadı: %s", target_path)
        raise RuntimeError(f"Kritik hata: {target_path} altında 'MLmodel' dosyası bulunamadı.")

    try:
        logger.info("Şampiyon XGBoost modeli diskten yükleniyor: %s", target_path)
        loaded_model = mlflow.xgboost.load_model(str(target_path))
        model_cache["model"] = loaded_model
        model_cache["model_path"] = str(target_path)
        model_cache["model_type"] = "XGBoostRegressor"
        logger.info("XGBoost modeli başarıyla belleğe alındı. Servis hazır.")
    except Exception as exc:
        logger.error("Model yükleme hatası: %s", str(exc), exc_info=True)
        raise RuntimeError(f"Servis başlatılamadı: {exc}")

    yield

    model_cache.clear()
    logger.info("Model bellekten tahliye edildi.")


app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="NASA C-MAPSS FD001 telemetri verisiyle motor Kalan Faydalı Ömür (RUL) kestirim servisi.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Kök dizine gelen istekleri Swagger arayüzüne yönlendirir."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Konteyner ve modelin sağlık durumunu raporlar."""
    is_ready = "model" in model_cache
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_type": model_cache.get("model_type", "unknown")
    }


@app.post("/predict", tags=["Inference"], response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    """Motor telemetri serisini alır, RUL kestirimi ve bakım aksiyonu üretir."""
    if "model" not in model_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model henüz hazır değil."
        )

    try:
        # 1. Ham telemetri geçmişini DataFrame'e dönüştür
        records = []
        for cycle in payload.history:
            cycle_dict = cycle.model_dump() if hasattr(cycle, "model_dump") else cycle.dict()
            row = {
                "unit_number": payload.unit_number,
                "time_in_cycles": cycle_dict.get("time_in_cycles", 1),
                "setting_1": cycle_dict.get("setting_1", 0.0),
                "setting_2": cycle_dict.get("setting_2", 0.0),
                "setting_3": cycle_dict.get("setting_3", 0.0),
            }
            row.update(cycle_dict.get("sensors", {}))
            records.append(row)

        df_input = pd.DataFrame(records).sort_values("time_in_cycles").reset_index(drop=True)
        current_cycle = int(df_input["time_in_cycles"].iloc[-1])

        # 2. Eğitimle aynı özellik mühendisliği fonksiyonu (81 özellik)[cite: 1, 2]
        df_features = generate_time_series_features(df_input, sensors=INFORMATIVE_SENSORS)

        # 3. Modelin beklediği sütun hizalaması
        model = model_cache["model"]
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            expected_cols = model.get_booster().feature_names
        else:
            expected_cols = [c for c in df_features.columns if c not in ["unit_number", "time_in_cycles"]]

        target_row = df_features.iloc[[-1]]
        X = target_row[expected_cols].copy()

        # 4. Model kestirimi ve clipping (maksimum 125 çevrim)[cite: 1, 2]
        raw_pred = model.predict(X)
        predicted_rul = float(np.clip(raw_pred[0], a_min=0.0, a_max=125.0))
        predicted_rul = round(predicted_rul, 2)

        # 5. Karar destek protokolü
        if predicted_rul <= 20:
            health_status = "CRITICAL"
            recommended_action = "Acil bakım planla; motor bir sonraki uçuştan önce hangara çekilmeli."
        elif predicted_rul <= 50:
            health_status = "WARNING"
            recommended_action = "Planlı bakımı gözden geçir, periyodik kontrol sıklığını artır."
        else:
            health_status = "HEALTHY"
            recommended_action = "Normal operasyon devam edebilir; telemetri değerleri nominal aralıkta."

        return PredictionResponse(
            unit_number=payload.unit_number,
            current_cycle=current_cycle,
            predicted_rul=predicted_rul,
            health_status=health_status,
            recommended_action=recommended_action
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Ö-05: İç hata ayrıntılarını istemciye sızdırmadan güvenli loglama[cite: 1, 2]
        logger.error("Tahmin sürecinde dahili hata oluştu: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tahmin hesaplanırken dahili bir sunucu hatası oluştu."
        )