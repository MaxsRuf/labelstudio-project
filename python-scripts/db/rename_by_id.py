import os
import shutil
import sqlite3

DB_PATH = "data/db/image_metadata.db"
IMAGES_DIR = "data/raw"
RENAMED_DIR = "data\processed"

def rename_files():
    if not os.path.exists(RENAMED_DIR):
        os.makedirs(RENAMED_DIR)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id, original_name, new_name FROM images")
    rows = c.fetchall()

    for id_, original_name, new_name in rows:
        old_path = os.path.join(IMAGES_DIR, original_name)
        new_path = os.path.join(RENAMED_DIR, new_name)

        if os.path.exists(old_path) and not os.path.exists(new_path):
            print(f"Копирование: {original_name} -> {new_name}")
            shutil.copy2(old_path, new_path)  # копируем с сохранением метаданных
            # Если надо переместить, используй shutil.move(old_path, new_path)
        else:
            print(f"Файл {original_name} не найден или {new_name} уже существует, пропускаем.")

    conn.close()

if __name__ == "__main__":
    rename_files()
