import pickle
from sentence_transformers import CrossEncoder
from src.embedding.embedding import EmbeddingModel
from src.vector_store.faiss_store import FAISSVectorStore
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import yaml

logger=get_logger(__name__)

def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Config file loaded successfully")
        return config

    except Exception as e:
        logger.error("Error loading config file")
        raise CustomException(e, sys)
class Retriever:
    def __init__(self):
        try:
            logger.info("Initializing Retriever")

            config = load_config()

            model_name=config['embedding']['model_name']
            self.embedding_model=EmbeddingModel(model_name=model_name)
            

            with open("artifacts/chunks.pkl", 'rb') as f:
                self.chunks=pickle.load(f)

            logger.info(f"Loaded {len(self.chunks)} chunks")

            self.vector_store=FAISSVectorStore(dimension=None)
            self.vector_store.load()

            dimension=self.vector_store.index.d
            logger.info(f"Detected FAISS dimension: {dimension}")

            retrieval_config=config['retrieval']

            self.top_k=retrieval_config.get("top_k", 3)
            reranker_model=retrieval_config.get(
                'reranker_model',
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

            self.reranker=CrossEncoder(reranker_model)
            logger.info(f"Reranker model loaded: {reranker_model}")

        except Exception as e:
            logger.error("Error initializing retriever")
            raise CustomException(e,sys)
        
    
    def retrieve(self,query, top_k=3):
        try:
            logger.info(f"Retrieving for query: {query}")

            query_embedding=self.embedding_model.encode([query])[0]
            distances,indices=self.vector_store.search(query_embedding, self.top_k)

            retrieved_chunks=[self.chunks[i] for i in indices[0]]

            logger.info(f"Initial retrieved chunks: {len(retrieved_chunks)}")

            pairs=[(query,chunk) for chunk in retrieved_chunks]
            scores=self.reranker.predict(pairs)

            ranked_chunks=[
                chunk for _, chunk in sorted(
                    zip(scores, retrieved_chunks),
                    key=lambda x: x[0],
                    reverse=True
                )
            ]

            filtered_chunks=[
                chunk for chunk in ranked_chunks
                if any(word.lower() in chunk.lower() for word in query.split())
            ]

            if not filtered_chunks:
                filtered_chunks=ranked_chunks

            logger.info("Chunks reranked and filtered successfully")

            return filtered_chunks[:self.top_k]
        
        except Exception as e:
            logger.error("Error during retrieval")
            raise CustomException(e,sys)
        

