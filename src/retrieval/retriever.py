import pickle
from src.embedding.embedding import EmbeddingModel
from src.vector_store.faiss_store import FAISSVectorStore
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

class Retriever:
    def __init__(self):
        try:
            logger.info("Initializing Retriever")

            self.embedding_model=EmbeddingModel()

            with open("artifacts/chunks.pkl", 'rb') as f:
                self.chunks=pickle.load(f)

            logger.info(f"Loaded {len(self.chunks)} chunks")

            self.vector_store=FAISSVectorStore(dimension=None)
            self.vector_store.load()

            dimension=self.vector_store.index.d
            logger.info(f"Detected FAISS dimension: {dimension}")

        except Exception as e:
            logger.error("Error initializing retriever")
            raise CustomException(e,sys)
        
    
    def retrieve(self,query, top_k=3):
        try:
            logger.info(f"Retrieving for query: {query}")

            query_embedding=self.embeddng_model.encode([query])[0]
            distances,indices=self.vector_store.search(query_embedding, top_k)
            res=[self.chunks[i] for i in indices[0]]

            return res
        
        except Exception as e:
            logger.error("Error during retrieval")
            raise CustomException(e,sys)
        

