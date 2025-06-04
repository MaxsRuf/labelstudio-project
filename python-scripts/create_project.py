import os
import logging
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

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

# Получение переменных
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")

if not LABEL_STUDIO_API_KEY or not LABEL_STUDIO_URL:
    logger.error("Не удалось загрузить LABEL_STUDIO_API_KEY или LABEL_STUDIO_URL из .env")
    exit(1)

def find_images(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    if not os.path.exists(directory):
        logger.warning(f"Директория {directory} не существует")
        return []
    return [f for f in os.listdir(directory) if any(f.lower().endswith(ext) for ext in image_extensions)]

def create_project_with_images():
    try:
        # Подключение
        client = LabelStudio(api_key=LABEL_STUDIO_API_KEY, base_url=LABEL_STUDIO_URL)
        logger.info(f"Подключение к Label Studio: {LABEL_STUDIO_URL}")

        # Конфигурация разметки
        label_config = '''
        <View>
            <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
            <RectangleLabels name="label" toName="image">
                <Label value="document" background="#FFA39E"/>

            </RectangleLabels>
        </View>
        '''

        # Создание проекта
        project = client.projects.create(title='Document Detection Project', label_config=label_config)
        logger.info(f"Проект создан: ID={project.id}, Title={project.title}")


        # Поиск изображений
        images_dir = "/shared-data/media/images"
        os.makedirs(images_dir, exist_ok=True)
        image_files = find_images(images_dir)

        if not image_files:
            logger.warning("Нет изображений для загрузки")
            return project.id

        # Добавление задач
        for filename in image_files:
            image_path = f"/data/local-files/?d=media/images/{filename}"
            # "image": "/data/local-files/?d=media/images/detected_lines.jpg"
            client.tasks.create(
                data={"image": image_path},
                project=project.id
            )
            logger.info(f"Задача создана для изображения: {filename}")

        logger.info(f"Все задачи добавлены. Проект доступен: {LABEL_STUDIO_URL}/projects/{project.id}/")
        return project.id

    except Exception as e:
        logger.error(f"Ошибка при создании проекта: {e}")
        return None

if __name__ == "__main__":
    project_id = create_project_with_images()
    if project_id:
        logger.info(f"Проект успешно создан с ID: {project_id}")
    else:
        logger.error("Не удалось создать проект")
