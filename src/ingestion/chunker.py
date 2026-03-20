from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

def chunk_text(text:str, chunk_size:int, overlap:int):
    try:
        logger.info("Starting text chunking")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        words=text.split()
        chunks=[]
        
        step=chunk_size-overlap

        for i in range(0, len(words), step):
            chunk=words[i:i + chunk_size]
            chunks.append(" ".join(chunk))

        logger.info(f"Total chunks created: {len(chunks)}")

        return chunks
    
    except Exception as e:
        logger.error("Error during chunking")
        raise CustomException(e,sys)