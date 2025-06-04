FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Создание директорий
RUN mkdir -p /app/models

# Копирование ML Backend кода
COPY ml_backend.py .

# Переменные окружения
ENV YOLO_MODEL_PATH=/app/models/yolo11x.pt
ENV YOLO_CONFIDENCE=0.25
ENV ML_BACKEND_PORT=9090
ENV PYTHONUNBUFFERED=1

# Открываем порт
EXPOSE 9090

# Команда запуска
CMD ["python", "ml_backend.py"]