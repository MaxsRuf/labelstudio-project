import os
import sqlite3
import pandas as pd

images_dir = "data/raw"
files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

# Извлекаем метаданные из имени (при необходимости)
data = []
for i, f in enumerate(files, start=1):
    row = {
        "id": i,
        "original_name": f,
        "new_name": f"{i:06d}" + os.path.splitext(f)[1],
        "angle": None,
        "type": None
    }

    # Пример разбора метаданных (если есть в имени)
    try:
        parts = f.split('_')
        row['angle'] = int(parts[2])
        row['type'] = parts[4].split('.')[0]
    except:
        pass

    data.append(row)

df = pd.DataFrame(data)

# Сохраняем в SQLite
try:
    conn = sqlite3.connect("data/db/image_metadata.db")
    df.to_sql("images", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("Database was created successfully✅")
except:
    print("DB create ERROR ❌ ")    

