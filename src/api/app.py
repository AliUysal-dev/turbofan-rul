import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

from src.api.schemas import PredictionRequest, PredictionResponse
from src.features import generate_time_series_features, INFORMATIVE_SENSORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger("turbofan_api")

model_cache: Dict[str, Any] = {}

def _resolve_model_directory() -> str:
    search_roots = [
        Path("mlruns"),
        Path("/app/mlruns"),
        Path(__file__).resolve().parents[2] / "mlruns",
        Path(__file__).resolve().parents[1] / "mlruns",
    ]
    for root in search_roots:
        if root.exists():
            mlmodel_candidates = list(root.rglob("MLmodel"))
            if mlmodel_candidates:
                chosen_dir = str(mlmodel_candidates[0].parent)
                logger.info("Model dizini bulundu: %s", chosen_dir)
                return chosen_dir
    raise FileNotFoundError("mlruns dizini altında 'MLmodel' bulunamadı.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Turbofan RUL API başlatılıyor...")
    try:
        model_dir = _resolve_model_directory()
        loaded_model = mlflow.xgboost.load_model(model_dir)
        model_cache["model"] = loaded_model
        model_cache["model_path"] = model_dir
        model_cache["run_id"] = Path(model_dir).parent.name
        logger.info("Model belleğe alındı.")
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
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    is_ready = "model" in model_cache
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_run_id": str(model_cache.get("run_id", "local_artifact"))
    }

@app.post("/predict", tags=["Inference"], response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    if "model" not in model_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model henüz hazır değil."
        )

    try:
        # 1. Ham veriyi DataFrame'e dönüştür
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

        # 2. En az 20 döngü kontrolü (READEME'de 21 yazıyor, siz hangisini isterseniz)
        #    window=20 olduğu için 20 yeterli, ama siz 21 diyorsanız 21 yapın.
        MIN_CYCLES = 20
        if len(df_input) < MIN_CYCLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En az {MIN_CYCLES} döngü geçmişi gönderilmelidir (gönderilen: {len(df_input)})."
            )

        # 3. Özellik mühendisliği (eğitimdeki ile aynı fonksiyon ve aynı sensör listesi)
        df_features = generate_time_series_features(df_input, sensors=INFORMATIVE_SENSORS)

        # 4. Modelin beklediği sütunları al
        model = model_cache["model"]
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        else:
            # fallback: modelden alınamazsa, tüm özellik sütunlarını kullan (risklidir)
            expected_cols = [c for c in df_features.columns if c not in ["unit_number", "time_in_cycles"]]

        # 5. Son satırı seç ve sadece beklenen sütunları al
        target_row = df_features.iloc[[-1]]
        X = target_row[expected_cols].copy()

        # 6. Tahmin
        raw_pred = model.predict(X)
        predicted_rul = float(np.clip(raw_pred[0], a_min=0.0, a_max=125.0))
        predicted_rul = round(predicted_rul, 2)

        # 7. Sağlık durumu (READEME ile uyumlu eşikler)
        if predicted_rul <= 20:
            health_status = "CRITICAL"
            recommended_action = "Acil bakım planla; motor bir sonraki uçuştan önce hangara çekilmeli."
        elif predicted_rul <= 50:
            health_status = "WARNING"
            recommended_action = "Planlı bakımı gözden geçir, yakın takip et."
        else:
            health_status = "HEALTHY"
            recommended_action = "Normal operasyon devam edebilir, rutin kontroller yeterli."

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
        logger.error("Tahmin hatası: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin hatası: {str(exc)}"
        )