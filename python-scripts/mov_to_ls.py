import os
import shutil

IMAGES_DIR = "data\processed"
LABEL_STUDIO_IMPORT_DIR = "data\labelstudio\media\images"

filles = os.listdir(LABEL_STUDIO_IMPORT_DIR)
for file in filles:
    print(str(os.path.join(LABEL_STUDIO_IMPORT_DIR,file)))
    file_path = os.path.join(LABEL_STUDIO_IMPORT_DIR,file)

    try:
        os.path.isfile(file_path)
    except:
        print("Это не файл")

    try:
        os.remove(file_path)
        print('Удаление удалось')
    except:
        print('Удаление не произошло')    

images = os.listdir(IMAGES_DIR)
for img in images:
    img_path = os.path.join(IMAGES_DIR, img)
    try:
        os.path.isfile(img_path)
    except:
        print("Это не файл")

    try:    
        shutil.copy2(str(img_path), str(LABEL_STUDIO_IMPORT_DIR))
        print("copy", img_path, "️✅")
    except:
        print("ошибка копирования")

