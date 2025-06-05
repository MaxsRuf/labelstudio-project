from flask import Flask, request, jsonify
import os
import logging
import numpy as np
from ultralytics import YOLO
import cv2
import base64
from PIL import Image
import io
import requests
from urllib.parse import urlparse, unquote
import torch

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class YOLOMLBackend:
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """Инициализация ML Backend с YOLO моделью"""
        self.confidence_threshold = confidence_threshold
        
        # Загрузка модели
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Загружаем пользовательскую модель: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info("Загружаем стандартную модель yolo11x")
                self.model = YOLO('yolo11x.pt')
                print(model_path)
                print(os.__file__)
                
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"YOLO будет использовать устройство: {device}")
            self.model.to(device)    
            logger.info("YOLO модель успешно загружена")
            
            # Получаем названия классов из модели
            self.class_names = getattr(self.model, 'names', {})
            logger.info(f"Классы модели: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def load_image_from_url(self, image_url):
        """Загрузка изображения по URL"""
        try:
            # Обработка локальных файлов Label Studio
            if '/data/local-files/?d=' in image_url:
                # Извлекаем путь к файлу
                file_path = image_url.split('?d=')[-1]
                file_path = unquote(file_path)
                
                # Полный путь к файлу в контейнере
                full_path = f"/shared-data/{file_path}"
                
                if os.path.exists(full_path):
                    image = cv2.imread(full_path)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
            # Обработка обычных HTTP URL
            elif image_url.startswith(('http://', 'https://')):
                response = requests.get(image_url)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    return np.array(image)
            
            # Обработка base64
            elif image_url.startswith('data:image'):
                header, data = image_url.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
                
            logger.error(f"Не удалось загрузить изображение: {image_url}")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {image_url}: {e}")
            return None

    def predict(self, image_url):
        """Предсказание для одного изображения"""
        try:
            # Загружаем изображение
            image = self.load_image_from_url(image_url)
            if image is None:
                return []

            # Получаем предсказания от YOLO
            results = self.model(image, conf=self.confidence_threshold, imgsz=1024)

            
            predictions = []
            
            for result in results:
                if result.boxes is not None:
                    img_height, img_width = result.orig_shape
                    
                    for box in result.boxes:
                        # Координаты в формате xyxy
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Название класса
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        # Преобразуем в проценты для Label Studio
                        x_percent = (x1 / img_width) * 100
                        y_percent = (y1 / img_height) * 100
                        width_percent = ((x2 - x1) / img_width) * 100
                        height_percent = ((y2 - y1) / img_height) * 100
                        
                        # Маппинг классов на метки Label Studio
                        label_mapping = {
                            'document': 'document'
                        }
                        
                        label = label_mapping.get(class_name.lower(), 'document')
                        
                        prediction = {
                            "from_name": "label",
                            "to_name": "image", 
                            "type": "rectanglelabels",
                            "value": {
                                "x": x_percent,
                                "y": y_percent,
                                "width": width_percent,
                                "height": height_percent,
                                "rectanglelabels": [label]
                            },
                            "score": confidence
                        }
                        
                        predictions.append(prediction)
            
            logger.info(f"Создано {len(predictions)} предсказаний для {image_url}")
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка предсказания для {image_url}: {e}")
            return []

# Глобальный объект ML Backend
ml_backend = None

def init_ml_backend():
    """Инициализация ML Backend"""
    global ml_backend
    
    model_path ="/app/models/model.pt"      
    print(model_path)
    confidence = float(os.getenv('YOLO_CONFIDENCE', '0.25'))
    
    logger.info(f"Инициализация ML Backend с моделью: {model_path}")
    ml_backend = YOLOMLBackend(model_path=model_path, confidence_threshold=confidence)

@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья сервиса"""
    return jsonify({"status": "healthy", "model_loaded": ml_backend is not None})

@app.route('/setup', methods=['POST'])
def setup():
    """Настройка ML Backend"""
    try:
        data = request.get_json()
        logger.info(f"Setup request: {data}")
        
        return jsonify({
            "status": "ok",
            "model_version": "YOLO11x Document Detection v1.0"
        })
    except Exception as e:
        logger.error(f"Ошибка setup: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Основной endpoint для предсказаний"""
    try:
        data = request.get_json()
        logger.info(f"Predict request received")
        
        if not ml_backend:
            return jsonify({"error": "ML Backend не инициализирован"}), 500
        
        # Извлекаем данные о задачах
        tasks = data.get('tasks', [])
        if not tasks:
            return jsonify({"results": []})
        
        results = []
        
        for task in tasks:
            task_data = task.get('data', {})
            image_url = task_data.get('image', '')
            
            if image_url:
                # Получаем предсказания от YOLO
                predictions = ml_backend.predict(image_url)
                
                result = {
                    "result": predictions,
                    "score": sum([p.get('score', 0) for p in predictions]) / len(predictions) if predictions else 0.0
                }
            else:
                result = {"result": [], "score": 0.0}
            
            results.append(result)
        
        response = {"results": results}
        logger.info(f"Возвращаем {len(results)} результатов")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Ошибка predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Endpoint для дообучения модели (заглушка)"""
    try:
        data = request.get_json()
        logger.info("Train request received (not implemented)")
        
        return jsonify({
            "status": "ok",
            "message": "Training not implemented for YOLO backend"
        })
        
    except Exception as e:
        logger.error(f"Ошибка train: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate():
    """Endpoint для валидации модели"""
    try:
        return jsonify({
            "status": "ok",
            "accuracy": 0.95  # Заглушка
        })
        
    except Exception as e:
        logger.error(f"Ошибка validate: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Инициализация ML Backend при запуске
    init_ml_backend()
    
    # Запуск Flask сервера
    port = int(os.getenv('ML_BACKEND_PORT', '9090'))
    app.run(host='0.0.0.0', port=port, debug=False)