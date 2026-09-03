
```markdown
# Turbofan Engine Remaining Useful Life (RUL) Prediction API

NASA C-MAPSS FD001 telemetri verileriyle eğitilmiş, endüstriyel standartlarda uçtan uca makine öğrenmesi ve kestirimci bakım servisidir. Servis; çok değişkenli sensör zaman serisi verilerinden kayan pencere özellikleri türetir, motorun kalan faydalı ömrünü (Remaining Useful Life) tahmin eder ve kural tabanlı karar destek protokolüyle bakım aksiyonu önerir.

---

## Canlı Servis Erişimi

Uygulama, Render altyapısı üzerinde Docker konteyneri olarak canlıda hizmet vermektedir. API uç noktalarını aşağıdaki bağlantılar üzerinden doğrudan test edebilirsiniz:

* Canlı API Dokümantasyonu (Swagger UI): https://turbofan-rul-api-ep09.onrender.com/docs
* Servis Canlılık Kontrolü: https://turbofan-rul-api-ep09.onrender.com/health

Not: Sistem sunucusuz ücretsiz katmanda (Free Tier) barındırılmaktadır. 15 dakika süresince istek almadığında uyku moduna geçer; ilk tetiklemede açılış süresi 30–40 saniye sürebilir.

---

## Mimari ve Tasarım Tercihleri

### 1. Özellik Mühendisliği (Feature Engineering)
* Kayan Pencere İstatistikleri: Bilgilendirici 14 sensör ve 2 operasyonel ayar (toplam 16 baz değişken) üzerinden motor bazında 5 ve 20 çevrimlik kayan ortalama (rolling mean) ile 5 çevrimlik standart sapma (rolling std) değerleri hesaplanır.
* Bozulma Trendi Tespiti: Anlık ölçümler ile 20 çevrimlik kayan ortalama arasındaki fark (delta_20) indeks hizalı olarak türetilerek aşınma ve anomali trendleri yakalanır.
* Toplam Özellik Hacmi: 16 değişken x 5 temsil (ham, roll_mean_5, roll_std_5, roll_mean_20, delta_20) + time_in_cycles olmak üzere model eğitiminde ve çıkarımında tam 81 özellik kullanılır.

### 2. Modelleme ve Deney Doğrulama
* Algoritma: Doğrusal olmayan yıpranma eğrilerini modellemek amacıyla optimize edilmiş XGBoost Regressor kullanılmıştır.
* Hedef Etiketleme (Piecewise Linear RUL): İlk işletme evrelerindeki aşınmasız durumu yansıtmak adına hedef RUL etiketi literatüre uygun şekilde maksimum 125 çevrimle sınırlandırılmıştır (clipping).
* Değerlendirme Metrikleri: Standart regresyon metriklerinin (RMSE, R2) yanı sıra, erken tahminleri az, geç tahminleri ise operasyonel risk nedeniyle katlanarak cezalandıran asimetrik NASA PHM 2008 skor fonksiyonu kullanılmıştır.

### 3. Model Karşılaştırma ve Doğrulama Kayıtları

MLflow takip sisteminde kayıtlı doğrulanabilir deney sonuçları:

| Model | Veri Bölmesi | RMSE | R2 | NASA PHM Skoru | Durum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Linear Regression) | Doğrulama (Validation) | 17.93 | 0.815 | - | Referans Baseline |
| XGBoost + Rolling Features | Test (FD001 Hold-out) | 16.52 | 0.830 | 475.87 | Üretim Modeli (Run: 6ccbe5e4) |

Not: Tablodaki değerler mlruns/ kayıt defterindeki gerçek çalıştırma çıktılarıyla birebir eşleşmektedir. Doğrulama bölmesi motor bazlı (unit-based group split) yapılmış, ardışık çevrimlerin sızması engellenmiştir.

### 4. Servis Mimarisi (FastAPI)
* Bellek Önbellekleme (Lifespan Context): Sabit üretim modeli servis ayağa kalkarken RAM'e bir defa alınır; disk okuma maliyeti sıfırlanarak milisaniye gecikme hedeflenir.
* Şema Doğrulama (Pydantic v2): Kayan pencere gereksinimi nedeniyle en az 20 döngü geçmişi (min_length=20) ve 14 zorunlu sensörün eksiksiz varlığı şema seviyesinde denetlenir.

---

## API Spesifikasyonu

### 1. Canlılık Kontrolü

* Yol: GET /health
* İşlev: Konteynerin aktifliğini ve modelin belleğe yüklendiğini doğrular.

Örnek Yanıt:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoostRegressor"
}

```

### 2. Kalan Faydalı Ömür Kestirimi

* Yol: POST /predict
* İşlev: Motorun geçmiş telemetri serisini alır, 81 özelliği türetir ve RUL kestirimi ile bakım aksiyonu üretir.

Örnek İstek Gövdesi:

```json
{
  "unit_number": 1,
  "history": [
    {
      "time_in_cycles": 1,
      "setting_1": -0.0007,
      "setting_2": -0.0004,
      "setting_3": 100.0,
      "sensors": {
        "sensor_2": 641.82,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_7": 554.36,
        "sensor_8": 2388.06,
        "sensor_9": 9046.19,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_17": 392.0,
        "sensor_20": 39.06,
        "sensor_21": 23.4190
      }
    }
  ]
}

```

Örnek Yanıt:

```json
{
  "unit_number": 1,
  "current_cycle": 20,
  "predicted_rul": 112.45,
  "health_status": "HEALTHY",
  "recommended_action": "Normal operasyon devam edebilir; telemetri değerleri nominal aralıkta."
}

```

---

## Karar Destek Protokolü

| Tahmin Edilen RUL | Durum Kodu | Önerilen Bakım Aksiyonu |
| --- | --- | --- |
| > 50 Çevrim | HEALTHY | Sensör değerleri nominal aralıkta; planlı uçuş operasyonu sürdürülür. |
| 21 – 50 Çevrim | WARNING | Aşınma trendi tespit edildi; periyodik kontrol sıklığı artırılır, planlı bakım listesine alınır. |
| 0 – 20 Çevrim | CRITICAL | Kritik eşik aşıldı; motor bir sonraki uçuştan önce hangara çekilerek acil bakıma alınır. |

---

## Yerel Kurulum ve Çalıştırma

### 1. Veri Setinin Temini (C-MAPSS FD001)

Eğitim boru hattını yerelde sıfırdan çalıştırmak için NASA C-MAPSS veri setini indirin:

1. NASA Prognostics Data Repository üzerinden C-MAPSS veri setini temin edin.
2. Aşağıdaki üç dosyayı projenin data/raw/ dizinine yerleştirin:
* train_FD001.txt
* test_FD001.txt
* RUL_FD001.txt



### 2. Docker ile Çalıştırma

```bash
# Repoyu klonlayın
git clone https://github.com/AliUysal-dev/turbofan-rul.git
cd turbofan-rul

# İmajı derleyin
docker build -t turbofan-rul:latest .

# Konteyneri başlatın
docker run -p 8000:8000 turbofan-rul:latest

```

Arayüze http://localhost:8000/docs adresinden erişebilirsiniz.

### 3. Python Ortamında Çalıştırma

```bash
# Sanal ortam
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / macOS

# Bağımlılıklar
pip install -r requirements.txt

# Modeli eğitme
python src/train.py

# Servisi başlatma
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Birim testlerini çalıştırma
python -m pytest tests/ -v

```

---

## Dizin Yapısı

```text
turbofan-rul/
├── Dockerfile                  # Konteyner derleme talimatı
├── requirements.txt            # Python kütüphane bağımlılıkları
├── README.md                   # Proje teknik dokümantasyonu
├── models/
│   └── production_model/       # Üretime mühürlenmiş XGBoost model ağırlıkları
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI giriş noktası, lifespan ve yönlendirme
│   │   └── schemas.py          # Pydantic v2 veri modelleri ve doğrulayıcılar
│   ├── features.py             # Kayan pencere ve delta özellik türetimi
│   ├── train.py                # Model eğitim boru hattı
│   └── config.py               # Proje sabitleri ve yol yapılandırmaları
├── tests/
│   └── test_features.py        # Özellik türetimi ve hizalama regresyon testleri
└── notebooks/                  # Keşifçi veri analizi not defterleri

```
