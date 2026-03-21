from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

# Dimension = 384 (for MiniLM)
class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model=SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error("Error loading embedding model")
            raise CustomException(e,sys)
        
    def encode(self, texts):
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings=self.model.encode(texts, show_progress_bar=True)
            return embeddings

        except Exception as e:
            logger.error("Error generating embeddings")
            raise CustomException(e,sys)
