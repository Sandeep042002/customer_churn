import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "mlproject"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_processing.py",
    f"src/{project_name}/components/exception.py",
    f"src/{project_name}/components/logger.py",
    f"src/{project_name}/components/utils.py",
    f"src/{project_name}/components/Data_training.py",
    f"src/{project_name}/components/constant.py",
    f"src/{project_name}/components/prediction.py",
    f"src/{project_name}/app.py",
    "main.py",
]

# Deduplicate file paths
list_of_files = list(set(list_of_files))

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"📁 Created directory: {filedir} for file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"📝 Created empty file: {filepath}")
    else:
        logging.info(f"✅ {filename} already exists")
