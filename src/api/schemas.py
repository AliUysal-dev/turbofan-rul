"""
src/api/schemas.py
Telemetri veri girişi ve RUL tahmin yanıtı için Pydantic veri modelleri.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class TelemetryCycle(BaseModel):
    """Tek bir uçuş/çalışma döngüsüne ait sensör ve ayar verisi."""
    time_in_cycles: int = Field(..., ge=1, description="Motor çalışma döngüsü")
    setting_1: float = Field(..., description="Çalışma ayarı 1")
    setting_2: float = Field(..., description="Çalışma ayarı 2")
    setting_3: float = Field(0.0, description="Çalışma ayarı 3")
    sensors: Dict[str, float] = Field(
        ..., 
        description="Sensör okumaları (örn: {'sensor_2': 642.1, 'sensor_4': 1400.5, ...})"
    )


class PredictionRequest(BaseModel):
    """Modelden tahmin istemek için gönderilen motor geçmiş telemetrisi."""
    unit_number: int = Field(..., ge=1, description="Motor seri / kimlik numarası")
    history: List[TelemetryCycle] = Field(
        ..., 
        min_length=1, 
        description="Motorun geçmiş telemetri listesi (en az 1, ideal olarak son 20 döngü)"
    )


class PredictionResponse(BaseModel):
    """API tarafından döndürülen kestirimci bakım kararı."""
    unit_number: int
    current_cycle: int
    predicted_rul: float = Field(..., description="Kalan Faydalı Ömür tahmini (döngü)")
    health_status: str = Field(..., description="Risk seviyesi: CRITICAL | WARNING | HEALTHY")
    recommended_action: str = Field(..., description="Operasyonel bakım aksiyonu")