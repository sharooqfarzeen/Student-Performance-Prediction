import logging
import os
import datetime from datetime

file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

file_path = os.path.join(os.getcwd(), "logs", file_name)

os.makedirs(file_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(file_path, file_name)

logging.basicConfig(
    file_name = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)