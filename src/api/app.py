from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

from src.utils.logger import logger
from src.api.schemas import PredictionRequest, PredictionResponse

# İsteğe bağlı şema ve konfigürasyon importları
try:
    from src.api.schemas import HealthResponse
except ImportError:
    HealthResponse = None

try:
    import src.config as config
except ImportError:
    config = None


# Global model önbelleği
model_cache: Dict[str, Any] = {}


def _resolve_model_directory() -> str:
    """
    Hem yerel Windows ortamında hem de Docker/Render Linux ortamında
    mlruns dizini altındaki MLmodel dosyasını bularak model dizinini döner.
    """
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
                # İlk bulunan geçerli model dizinini seç
                chosen_dir = str(mlmodel_candidates[0].parent)
                logger.info("Fiziksel model dizini tespit edildi: %s", chosen_dir)
                return chosen_dir

    raise FileNotFoundError("mlruns dizininde geçerli bir 'MLmodel' dosyası bulunamadı.")


def _engineer_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    src.features modülündeki fonksiyonları çağırır; fonksiyon bulunamazsa
    aynı zaman serisi kayan pencere (rolling) özelliklerini türetir.
    """
    try:
        import src.features as feats
        for fn_name in ["engineer_features", "create_features", "add_features", "calculate_features", "prepare_features"]:
            if hasattr(feats, fn_name):
                return getattr(feats, fn_name)(df)
    except Exception as exc:
        logger.warning("src.features çağrısında hata, yerel hesaplayıcıya geçiliyor: %s", exc)

    # Fallback: Kayan pencere ve türev hesaplamaları
    df_out = df.copy()
    sensor_cols = [c for c in df_out.columns if c.startswith("sensor_")]

    for s in sensor_cols:
        df_out[f"{s}_rolling_mean_5"] = df_out[s].rolling(window=5, min_periods=1).mean()
        df_out[f"{s}_rolling_mean_20"] = df_out[s].rolling(window=20, min_periods=1).mean()
        df_out[f"{s}_rolling_std_5"] = df_out[s].rolling(window=5, min_periods=1).std().fillna(0.0)
        df_out[f"{s}_delta_20"] = df_out[s] - df_out[f"{s}_rolling_mean_20"]

    return df_out


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI yaşam döngüsü yöneticisi: Servis ayağa kalkarken modeli belleğe alır,
    kapanırken kaynakları temizler.
    """
    logger.info("Turbofan RUL API başlatılıyor. Model diski taranıyor...")
    
    try:
        model_dir = _resolve_model_directory()
        logger.info("Model yükleniyor: %s", model_dir)
        
        loaded_model = mlflow.xgboost.load_model(model_dir)
        model_cache["model"] = loaded_model
        model_cache["model_path"] = model_dir
        model_cache["run_id"] = Path(model_dir).parent.name
        
        logger.info("XGBoost modeli başarıyla belleğe alındı. Servis hazır.")
    except Exception as exc:
        logger.error("Model yükleme aşamasında kritik hata: %s", str(exc), exc_info=True)
        raise RuntimeError(f"Servis başlatılamadı: {exc}")

    yield

    model_cache.clear()
    logger.info("Model bellekten temizlendi, servis güvenli şekilde durduruldu.")


# FastAPI Uygulama Tanımı
app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="NASA C-MAPSS FD001 telemetri verisiyle motor Kalan Faydalı Ömür (RUL) kestirim servisi.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Kök dizine gelen kullanıcıları doğrudan Swagger dokümantasyonuna yönlendirir."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"], response_model=HealthResponse if HealthResponse else None)
async def health_check():
    """Servis ve model yükleme durumunu doğrulayan uç nokta."""
    is_ready = "model" in model_cache
    payload = {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_run_id": str(model_cache.get("run_id", "local_artifact"))
    }
    return payload


@app.post("/predict", tags=["Inference"], response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    """
    Motor geçmiş telemetri verisini alır, özellik mühendisliği uygular
    ve kalan faydalı ömür kestirimi ile bakım kararı üretir.
    """
    if "model" not in model_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model henüz belleğe yüklenmedi veya servis hazırlık aşamasında."
        )

    try:
        # 1. Gelen geçmiş telemetri serisini DataFrame formatına çevir
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

        df_input = pd.DataFrame(records)
        df_input = df_input.sort_values("time_in_cycles").reset_index(drop=True)
        current_cycle = int(df_input["time_in_cycles"].iloc[-1])

        # 2. Özellik dönüşümlerini uygula
        df_features = _engineer_features_safe(df_input)
        target_row = df_features.iloc[[-1]].copy()

        # 3. Modelin beklediği sütunları hizala
        model = model_cache["model"]
        expected_cols = None

        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            expected_cols = model.get_booster().feature_names

        if expected_cols:
            for col in expected_cols:
                if col not in target_row.columns:
                    target_row[col] = 0.0
            X = target_row[expected_cols]
        else:
            drop_candidates = ["unit_number", "time_in_cycles", "RUL"]
            X = target_row.drop(columns=[c for c in drop_candidates if c in target_row.columns])

        # 4. Model tahmini ve clipping (RUL max 125)
        raw_pred = model.predict(X)
        predicted_rul = float(np.clip(raw_pred[0], a_min=0.0, a_max=125.0))
        predicted_rul = round(predicted_rul, 2)

        # 5. Karar destek mantığı
        if predicted_rul <= 15:
            health_status = "CRITICAL"
            recommended_action = "Acil bakım planla"
        elif predicted_rul <= 45:
            health_status = "WARNING"
            recommended_action = "Planlı bakıma al"
        else:
            health_status = "HEALTHY"
            recommended_action = "Normal operasyon devam edebilir"

        return PredictionResponse(
            unit_number=payload.unit_number,
            current_cycle=current_cycle,
            predicted_rul=predicted_rul,
            health_status=health_status,
            recommended_action=recommended_action
        )

    except Exception as exc:
        logger.error("Tahmin üretimi sırasında istisna oluştu: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin üretilemedi: {str(exc)}"
        )