import faiss
import numpy as np
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import os

logger=get_logger(__name__)

class FAISSVectorStore:
    def __init__(self,dimension):
        try:
            logger.info(f"Initializing FAISS index with dimension: {dimension}")
            self.index=faiss.IndexFlatL2(dimension)
        
        except Exception as e:
            logger.error("Error initializing FAISS")
            raise CustomException(e,sys)
        
    def add_embeddings(self, embeddings):
        try:
            logger.info(f"Adding {len(embeddings)} embeddings to FAISS")
            self.index.add(np.array(embeddings).astype('float32'))

        except Exception as e:
            logger.error("Error adding embeddings to FAISS")
            raise CustomException(e,sys)
        
    def search(self, query_embedding, top_k=5):
        try:
            logger.info("Performing similarity search")
            d,i=self.index.search(np.array([query_embedding]).astype("float32"), top_k)
            return d,i
        
        except Exception as e:
            logger.error("Error during search")
            raise CustomException(e,sys)
        
    def save(self, path="artifacts/faiss_index/index.faiss"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            faiss.write_index(self.index, path)
            logger.info(f"FAISS index saved at {path}")

        except Exception as e:
            logger.error("Error saving FAISS index")
            raise CustomException(e,sys)
        
    def load(self,path="artifacts/faiss_index/index.faiss"):
        try:
            self.index=faiss.read_index(path)
            logger.info("FAISS index loaded")

        except Exception as e:
            logger.error("Error loading FAISS index")
            raise CustomException(e,sys)