import os
import logging
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path)

LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")

if not LABEL_STUDIO_API_KEY or not LABEL_STUDIO_URL:
    logger.error("Не удалось загрузить переменные окружения")
    exit(1)

class YOLOAutoLabeler:
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Инициализация автоматической разметки с YOLO
        
        Args:
            model_path: путь к файлу модели .pt (если None, будет использована yolo11x)
            confidence_threshold: порог уверенности для детекций
        """
        self.client = LabelStudio(api_key=LABEL_STUDIO_API_KEY, base_url=LABEL_STUDIO_URL)
        self.confidence_threshold = confidence_threshold
        
        # Загрузка модели YOLO
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Загружаем пользовательскую модель: {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info("Загружаем стандартную модель yolo11x")
                self.model = YOLO('model.pt')  # будет скачана автоматически
                
            logger.info("Модель YOLO успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели YOLO: {e}")
            raise
    
    def detect_documents(self, image_path):
        """
        Обнаружение документов на изображении
        
        Args:
            image_path: путь к изображению
            
        Returns:
            list: список обнаруженных объектов с координатами и уверенностью
        """
        try:
            # Предсказание с помощью YOLO
            results = self.model(image_path, conf=self.confidence_threshold)
            
            detections = []
            
            for result in results:
                # Получаем размеры изображения
                img_height, img_width = result.orig_shape
                
                # Обрабатываем каждое обнаружение
                if result.boxes is not None:
                    for box in result.boxes:
                        # Координаты в формате xyxy
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Получаем название класса
                        class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"class_{class_id}"
                        
                        # Конвертируем в проценты для Label Studio
                        x_percent = (x1 / img_width) * 100
                        y_percent = (y1 / img_height) * 100
                        width_percent = ((x2 - x1) / img_width) * 100
                        height_percent = ((y2 - y1) / img_height) * 100
                        
                        detection = {
                            'x': x_percent,
                            'y': y_percent,
                            'width': width_percent,
                            'height': height_percent,
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        }
                        
                        detections.append(detection)
                        logger.debug(f"Обнаружен {class_name}: уверенность {confidence:.3f}")
            
            logger.info(f"Обнаружено объектов: {len(detections)}")
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_path}: {e}")
            return []
    
    def create_label_studio_annotation(self, detections, task_id):
        """
        Создание аннотации в формате Label Studio
        
        Args:
            detections: список обнаружений от YOLO
            task_id: ID задачи в Label Studio
            
        Returns:
            dict: аннотация в формате Label Studio
        """
        results = []
        
        for detection in detections:
            # Маппинг классов YOLO на метки Label Studio
            class_mapping = {
                'document': 'document',
                'paper': 'document',
                'page': 'document',
                'text': 'text',
                'table': 'table',
                'figure': 'figure',
                'image': 'figure'
            }
            
            # Определяем метку для Label Studio
            label_name = class_mapping.get(detection['class_name'].lower(), 'document')
            
            result = {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": detection['x'],
                    "y": detection['y'],
                    "width": detection['width'],
                    "height": detection['height'],
                    "rectanglelabels": [label_name]
                }
            }
            results.append(result)
        
        annotation = {
            "task": task_id,
            "result": results,
            "score": sum([d['confidence'] for d in detections]) / len(detections) if detections else 0.0
        }
        
        return annotation
    
    def process_project_tasks(self, project_id, max_tasks=None):
        """
        Обработка всех неразмеченных задач в проекте
        
        Args:
            project_id: ID проекта в Label Studio
            max_tasks: максимальное количество задач для обработки (None = все)
        """
        try:
            logger.info(f"Начинаем обработку проекта {project_id}")
            
            # Получаем все задачи проекта
            tasks = self.client.tasks.list(project=project_id)
            logger.info(f"Найдено задач в проекте: {len(tasks)}")
            
            # Фильтруем неразмеченные задачи (без аннотаций)
            unlabeled_tasks = [task for task in tasks if not task.get('annotations')]
            logger.info(f"Неразмеченных задач: {len(unlabeled_tasks)}")
            
            if max_tasks:
                unlabeled_tasks = unlabeled_tasks[:max_tasks]
                logger.info(f"Обрабатываем первые {len(unlabeled_tasks)} задач")
            
            processed_count = 0
            
            for task in unlabeled_tasks:
                try:
                    task_id = task['id']
                    image_url = task['data'].get('image', '')
                    
                    logger.info(f"Обрабатываем задачу {task_id}: {image_url}")
                    
                    # Извлекаем имя файла из URL
                    if '/data/local-files/?d=images/' in image_url:
                        filename = image_url.split('/data/local-files/?d=images/')[-1]
                        # Декодируем URL если нужно
                        from urllib.parse import unquote
                        filename = unquote(filename)
                        
                        image_path = f"/shared-data/images/{filename}"
                        
                        if os.path.exists(image_path):
                            # Обнаруживаем документы с помощью YOLO
                            detections = self.detect_documents(image_path)
                            
                            if detections:
                                # Создаем аннотацию для Label Studio
                                annotation = self.create_label_studio_annotation(detections, task_id)
                                
                                # Создаем предсказание в Label Studio
                                self.client.predictions.create(
                                    task=task_id,
                                    result=annotation['result'],
                                    score=annotation['score']
                                )
                                
                                logger.info(f"Создано предсказание для задачи {task_id} с {len(detections)} объектами")
                                processed_count += 1
                            else:
                                logger.warning(f"Не обнаружено объектов в задаче {task_id}")
                        else:
                            logger.warning(f"Файл не найден: {image_path}")
                    else:
                        logger.warning(f"Неподдерживаемый формат URL: {image_url}")
                        
                except Exception as e:
                    logger.error(f"Ошибка обработки задачи {task_id}: {e}")
                    continue
            
            logger.info(f"Обработано задач: {processed_count}")
            return processed_count
            
        except Exception as e:
            logger.error(f"Ошибка при обработке проекта: {e}")
            return 0
    
    def process_single_task(self, task_id):
        """
        Обработка одной конкретной задачи
        
        Args:
            task_id: ID задачи
        """
        try:
            task = self.client.tasks.get(id=task_id)
            image_url = task.data.get('image', '')
            
            if '/data/local-files/?d=images/' in image_url:
                filename = image_url.split('/data/local-files/?d=images/')[-1]
                from urllib.parse import unquote
                filename = unquote(filename)
                
                image_path = f"/shared-data/images/{filename}"
                
                if os.path.exists(image_path):
                    detections = self.detect_documents(image_path)
                    
                    if detections:
                        annotation = self.create_label_studio_annotation(detections, task_id)
                        
                        self.client.predictions.create(
                            task=task_id,
                            result=annotation['result'],
                            score=annotation['score']
                        )
                        
                        logger.info(f"Предсказание создано для задачи {task_id}")
                        return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Ошибка обработки задачи {task_id}: {e}")
            return False

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Автоматическая разметка с YOLO11x')
    parser.add_argument('--project-id', type=int, required=True, help='ID проекта в Label Studio')
    parser.add_argument('--model-path', type=str, help='Путь к файлу модели .pt')
    parser.add_argument('--confidence', type=float, default=0.25, help='Порог уверенности (по умолчанию 0.25)')
    parser.add_argument('--max-tasks', type=int, help='Максимальное количество задач для обработки')
    parser.add_argument('--task-id', type=int, help='ID конкретной задачи для обработки')
    
    args = parser.parse_args()
    
    try:
        # Инициализация автоматической разметки
        auto_labeler = YOLOAutoLabeler(
            model_path=args.model_path,
            confidence_threshold=args.confidence
        )
        
        if args.task_id:
            # Обработка одной задачи
            logger.info(f"Обрабатываем задачу {args.task_id}")
            success = auto_labeler.process_single_task(args.task_id)
            if success:
                logger.info("Задача успешно обработана")
            else:
                logger.error("Не удалось обработать задачу")
        else:
            # Обработка всего проекта
            processed = auto_labeler.process_project_tasks(
                project_id=args.project_id,
                max_tasks=args.max_tasks
            )
            logger.info(f"Обработано задач: {processed}")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()