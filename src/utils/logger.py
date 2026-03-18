import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name):
    return logging.getLogger(name)