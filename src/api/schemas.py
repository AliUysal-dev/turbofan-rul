from typing import Dict, List, Literal
from pydantic import BaseModel, Field, field_validator

# Modelin zorunlu kıldığı 14 bilgilendirici sensör listesi
REQUIRED_SENSORS = {
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
}


class CycleTelemetry(BaseModel):
    """Tek bir uçuş/çalışma döngüsüne ait operasyonel ayarlar ve sensör okumaları."""
    time_in_cycles: int = Field(..., ge=1, description="Motorun çalışma döngüsü (cycle) indeksi.")
    setting_1: float = Field(0.0, description="Operasyonel ayar 1.")
    setting_2: float = Field(0.0, description="Operasyonel ayar 2.")
    setting_3: float = Field(100.0, description="Operasyonel ayar 3.")
    sensors: Dict[str, float] = Field(
        ...,
        description="Sensör telemetri sözlüğü (14 zorunlu sensörü içermelidir)."
    )

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ö-04 çözümü: 14 zorunlu sensörün varlığını doğrular, eksikleri raporlar."""
        missing = REQUIRED_SENSORS - set(v.keys())
        if missing:
            missing_sorted = sorted(list(missing))
            raise ValueError(f"Eksik sensör telemetrisi tespit edildi: {missing_sorted}")
        return v


class PredictionRequest(BaseModel):
    """RUL kestirimi için API'ye iletilen telemetri serisi."""
    unit_number: int = Field(..., ge=1, description="Motor seri/ünite kimlik numarası.")
    history: List[CycleTelemetry] = Field(
        ...,
        min_length=20,
        description="Ö-03 çözümü: Kayan pencere (rolling) hesaplamaları için en az 20 döngü geçmişi zorunludur."
    )


class PredictionResponse(BaseModel):
    """Model kestirimi ve karar destek yanıtı."""
    unit_number: int
    current_cycle: int
    predicted_rul: float = Field(..., description="Tahmin edilen kalan faydalı ömür (çevrim).")
    health_status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    recommended_action: str