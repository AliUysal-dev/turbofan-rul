# Turbofan Engine RUL Estimation with XGBoost
### End-to-End ML Pipeline with FastAPI & MLflow

[![Python](https://img.shields.io/badge/Python-3.11-1f425f.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-005571.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/Lifecycle-MLflow-0194E2.svg)](https://mlflow.org/)
[![Model](https://img.shields.io/badge/Model-XGBoost_Regressor-black.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)

Bu depo, ticari turbofan uçak motorlarının Kalan Faydalı Ömür (**Remaining Useful Life - RUL**) kestirimini gerçekleştiren örnek bir uçtan uca makine öğrenmesi uygulamasıdır. Proje, NASA Prognostics Center of Excellence tarafından yayımlanan **C-MAPSS (FD001)** telemetri verisi üzerinde doğrulanmış olup; özellik mühendisliği, model yaşam döngüsü takibi ve üretim çıkarım servisini (Inference API) modüler bir yazılım mimarisinde birleştirir.

**Canlı Demo:** [https://turbofan-rul-api-ep09.onrender.com](https://turbofan-rul-api-ep09.onrender.com)  
*Not: Render ücretsiz servisi nedeniyle ilk istek 30-50 saniye gecikebilir.*

---

## 1. Sistem Mimarisi ve Tasarım İlkeleri

Sistem, veri bilimi prototipleri ile kurumsal yazılım mühendisliği arasındaki ayrımı netleştirmek üzere dört katmanlı bir yapıda tasarlanmıştır:

```text
[ C-MAPSS Raw Stream ]
         │
         ▼
[ Feature Pipeline ] ──> Dynamic Rolling (w=5, 20) & Delta Extraction (61 Features)
         │
         ▼
[ XGBoost Estimator ] ──> Piecewise Linear Degradation Modeling (RUL_max = 125)
         │
         ▼
[ Artifact Registry ] ──> MLflow Run Tracking & Model Serialization
         │
         ▼
[ Inference Service ] ──> FastAPI Lifespan Engine -> Decision Engine (Healthy/Warning/Critical)
```

### Temel Mühendislik Kararları
* **Yapılandırılmış Günlükleme (Logging):** Betik içi standart `print` ifadeleri elenmiş; zaman damgası, log seviyesi (`INFO`, `ERROR`) ve modül izi sağlayan merkezi bir logger entegre edilmiştir.
* **Tip Güvenliği ve Şema Doğrulama:** API giriş/çıkış sözleşmeleri `Pydantic v2` veri sınıfları üzerinden sıkı biçimde doğrulanır.
* **Bağımsız Konfigürasyon:** Veri yolları, pencere parametreleri ve model hiperparametreleri merkezi `config.py` modülünde toplanmıştır.

---

## 2. Metodoloji ve Modelleme Stratejisi

### Hedef Değişken Modellemesi (Piecewise Linear Target)
Turbofan motorlarındaki mekanik aşınma ilk çalışma döngülerinde ihmal edilebilir düzeydedir. Bu fiziksel gerçeği modele yansıtmak amacıyla erken döngülerdeki hedef değişken $RUL_{max} = 125$ seviyesinde sabitlenmiştir:

$$RUL_{target}(t) = \min(RUL_{actual}(t), 125)$$

### Özellik Mühendisliği (Feature Engineering)
* **Sabit Sinyal Eliminasyonu:** Operasyonel koşullar altında varyansı sıfır olan 8 sensör ve ayar değişkeni çıkarılmıştır.
* **Zaman Serisi Türetimleri:** Kalan 14 bilgilendirici sensör üzerinden motor bazlı 5 ve 20 döngülük kayan ortalamalar (`rolling mean`), kısa vadeli dalgalanmaları yakalamak için 5 döngülük standart sapmalar (`rolling std`) ve uzun vadeli aşınmayı modellemek için 20 döngülük farklar (`delta`) hesaplanmıştır.

### Değerlendirme Metrikleri
Sistem yalnızca simetrik metriklerle (RMSE, MAE) değil, havacılık endüstrisi standardı olan **NASA PHM 2008 Asimetrik Ceza Fonksiyonu** ile değerlendirilir. Bu fonksiyonda geciken tahminler ($d > 0$), erken tahminlere ($d < 0$) kıyasla üssel olarak daha sert cezalandırılır:

$$d = \hat{y} - y$$

$$S = \sum_{i=1}^{N} \begin{cases} e^{-\frac{d_i}{13}} - 1 & \text{için } d_i < 0 \\ e^{\frac{d_i}{10}} - 1 & \text{için } d_i \ge 0 \end{cases}$$

---

## 3. Deneysel Sonuçlar ve Karşılaştırmalı Başarım

Model performansı, NASA'nın eğitim sürecinde modele gösterilmeyen 100 test motorunun son döngüleri üzerinde test edilmiştir:

| Model ve Konfigürasyon | Test RMSE | Test MAE | Test $R^2$ | NASA PHM Skoru |
|---|---|---|---|---|
| Lineer Regresyon (Baseline) | 29.40 | 22.10 | 0.5120 | 2,850.40 |
| Random Forest (Ham Sensörler) | 20.15 | 14.80 | 0.7250 | 1,120.30 |
| LightGBM (Ham Sensörler) | 19.85 | 14.20 | 0.7410 | 980.15 |
| **XGBoost + Dynamic Time-Series Features (Bu Çalışma)** | **16.52** | **11.56** | **0.8300** | **475.87** |

---

## 4. Dizin Yapısı

```text
turbofan-rul/
├── data/
│   └── raw/                    # C-MAPSS FD001 ham veri dosyaları
├── mlruns/                     # MLflow artifact ve model deposu
├── notebooks/                  # Veri keşfi ve prototipleme
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_advanced_modeling.ipynb
├── src/                        # Üretim kodu
│   ├── api/
│   │   ├── app.py              # FastAPI servis katmanı ve lifespan yöneticisi
│   │   └── schemas.py          # Pydantic v2 veri sözleşmeleri
│   ├── utils/
│   │   └── logger.py           # Yapılandırılmış loglama altyapısı
│   ├── config.py               # Merkezi parametre ve yol yönetimi
│   ├── features.py             # Zaman serisi dönüşüm fonksiyonları
│   ├── metrics.py              # Değerlendirme ve ceza metrikleri
│   └── train.py                # Pipeline orkestrasyon ve eğitim sınıfı
├── Dockerfile                  # Konteyner dağıtım konfigürasyonu
├── .dockerignore
├── requirements.txt            # Python bağımlılıkları
└── README.md
```

---

## 5. Kurulum ve Çalıştırma

### Gereksinimler
* Python 3.11+
* Conda veya Python Sanal Ortamı

### 1. Ortamın Hazırlanması
```bash
git clone https://github.com/AliUysal-dev/turbofan-rul.git
cd turbofan-rul
conda create -n turbofan python=3.11 -y
conda activate turbofan
pip install -r requirements.txt
```

### 2. Model Eğitimi ve MLflow Kaydı
Eğitim boru hattı ham veriyi işler, zaman serisi özelliklerini türetir, XGBoost modelini eğitir ve tüm sonuçları yerel MLflow deposuna mühürler:
```bash
python src/train.py
```

### 3. API Servisinin Başlatılması
Modeli belleğe alan ve REST uç noktası sunan FastAPI servisi:
```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Etkileşimli OpenAPI dokümantasyonu: `http://127.0.0.1:8000/docs`

Canlı ortamda etkileşimli dokümantasyon: [https://turbofan-rul-api-ep09.onrender.com/docs](https://turbofan-rul-api-ep09.onrender.com/docs)

### 4. Konteyner Ortamı (Docker)
Servisi izole bir Docker konteynerinde derlemek ve çalıştırmak için:
```bash
docker build -t turbofan-rul:latest .
docker run -p 8000:8000 turbofan-rul:latest
```

---

## 6. API Arayüzü ve Veri Sözleşmesi

**Endpoint:** `POST /predict`  
**İçerik Türü:** `application/json`  
**Canlı Uç Nokta:** `https://turbofan-rul-api-ep09.onrender.com/predict`

> **Önemli:** Kayan pencere özelliklerinin (window=5 ve window=20) hesaplanabilmesi için `history` dizisi **en az 21 zaman adımı (cycle)** içermelidir. Aksi takdirde API hata döndürecektir.

### İstek Formatı (Request Payload)
```json
{
  "unit_number": 42,
  "history": [
    {
      "time_in_cycles": 260,
      "setting_1": 0.0021,
      "setting_2": 0.0003,
      "setting_3": 100.0,
      "sensors": {
        "sensor_2": 643.90, "sensor_3": 1602.40, "sensor_4": 1428.80,
        "sensor_7": 551.20, "sensor_8": 2388.22, "sensor_9": 9070.10,
        "sensor_11": 48.15, "sensor_12": 520.10, "sensor_13": 2388.24,
        "sensor_14": 8145.20, "sensor_15": 8.5120, "sensor_17": 395,
        "sensor_20": 38.45, "sensor_21": 23.1200
      }
    }
  ]
}
```

### Yanıt Formatı (Response Payload)
```json
{
  "unit_number": 42,
  "current_cycle": 260,
  "predicted_rul": 15.81,
  "health_status": "CRITICAL",
  "recommended_action": "Acil bakım planla; motor bir sonraki uçuştan önce hangara çekilmeli."
}
```

**Sağlık Durumu Eşik Değerleri:**
- `HEALTHY` → Tahmini RUL > 50 döngü
- `WARNING` → 20 < Tahmini RUL ≤ 50 döngü
- `CRITICAL` → Tahmini RUL ≤ 20 döngü
  
