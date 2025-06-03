import os
import logging
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.create import choices

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,  # Можно поменять на DEBUG, если нужно больше подробностей
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

logger.info(f"Подключение к Label Studio на {LABEL_STUDIO_URL}")

def get_image_files(directory):
    """Получить список файлов изображений из директории"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    if not os.path.exists(directory):
        logger.warning(f"Директория {directory} не существует")
        return image_files
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    logger.info(f"Найдено {len(image_files)} изображений в {directory}")
    return image_files

def upload_images_to_project(project, images_directory, relative_path="test_data"):
    """Загрузить изображения в проект Label Studio"""
    image_files = get_image_files(images_directory)
    
    if not image_files:
        logger.warning("Нет изображений для загрузки")
        return
    
    tasks = []
    for filename in image_files:
        # Создаем путь для Label Studio (относительно /label-studio/data/)
        image_path = f"/data/local-files/?d={relative_path}/{filename}"
        
        task_data = {
            "data": {
                "image": image_path
            }
        }
        tasks.append(task_data)
        logger.info(f"Подготовлено к загрузке: {filename}")
    
    try:
        # Импорт задач в проект
        import_result = project.import_tasks(tasks)
        logger.info(f"Успешно загружено {len(tasks)} изображений в проект")
        return import_result
    except Exception as e:
        logger.error(f"Ошибка при загрузке изображений: {e}")
        return None

try:
    # Подключение к Label Studio
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    
    # Создание интерфейса разметки
    label_config = '''
    <View>
        <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
        <RectangleLabels name="label" toName="image">
            <Label value="document" background="#FFA39E"/>
            <Label value="text" background="#D4EDDA"/>
            <Label value="table" background="#CCE5FF"/>
            <Label value="figure" background="#FFE6CC"/>
        </RectangleLabels>
    </View>
    '''
    
    # Создание проекта
    project = ls.projects.create(
        title='Document Detection Project',
        label_config=label_config
    )
    
    logger.info(f"Проект создан: ID={project.id}, Title={project.title}")
    
    # Определяем путь к изображениям
    # Путь в контейнере python-scripts (shared-data смонтирована в /shared-data)
    images_directory = "/shared-data/images"
    relative_path = "images"
    
    # Создаем директорию images если её нет
    os.makedirs(images_directory, exist_ok=True)
    
    # Загружаем изображения
    logger.info(f"Поиск изображений в директории: {images_directory}")
    result = upload_images_to_project(project, images_directory, relative_path)
    
    if not result:
        logger.warning("Изображения не были загружены.")
        logger.warning("Поместите изображения в папку: data/labelstudio/images/")
        logger.info(f"Пример команды: cp /path/to/your/photos/* data/labelstudio/images/")
    
    logger.info(f"Проект доступен по адресу: {LABEL_STUDIO_URL}/projects/{project.id}/")

except Exception as e:
    logger.exception("Ошибка при подключении или создании проекта в Label Studio:")
    exit(1)