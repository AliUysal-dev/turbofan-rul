# 1. Temel Python imajı
FROM python:3.11-slim

# 2. Konteyner içi çalışma dizini
WORKDIR /app

# 3. Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Servis kodlarını ve eğitilmiş MLflow modelini kopyala
COPY src/ ./src/
COPY models/ ./models/

# 5. API portunu dışarı aç
EXPOSE 8000

# 6. Servisi başlat
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]