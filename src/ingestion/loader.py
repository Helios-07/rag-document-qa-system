import os
import fitz
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

def load_pdf(file_path):
    try:
        doc=fitz.open(file_path)
        text=""

        for page in doc:
            text+=page.get_text()

        return text
    
    except Exception as e:
        logger.error("Error extracting text using PyMuPDF")
        raise e

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document(file_path: str)-> str:
    try:
        logger.info(f"Loding document: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        
        ext = os.path.splitext(file_path)[1].lower().strip()
        ext = ext.replace(".", "")
        logger.info(f"Detected extension: {ext}")

        if ext=="txt":
            return load_text(file_path)
        elif ext=="pdf":
            return load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
    except Exception as e:
        logger.error("Errpr in load_document")
        raise CustomException(e,sys)