from src.retrieval.retriever import Retriever
from src.generation.generation import Generator
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import yaml

logger=get_logger(__name__)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class RAGPipeline:
    def __init__(self):
        try:
            logger.info("Initializing RAG Pipeline")

            config=load_config()

            self.retriever=Retriever()

            gen_config=config['generation']
            self.generator=Generator(
                model_name=gen_config['model_name'],
                max_length=gen_config.get('max_length', 200)
            )

        except Exception as e:
            logger.error("Error initializing RAG pipeline")
            raise CustomException(e,sys)
        
    
    def run(self,query):
        try:
            logger.info(f"Running RAG pipeline for query: {query}")

            retrieved_chunks=self.retriever.retrieve(query, top_k=2)
            retrieved_chunks = [chunk for chunk in retrieved_chunks if len(chunk) > 50]

            clean_chunks=[chunk.replace("\n", " ") for chunk in retrieved_chunks if len(chunk.strip())>50]
            logger.info(f"Retrieved {len(clean_chunks)} valid chunks")

            context = "\n\n".join(chunk[:400] for chunk in clean_chunks[:2])
            logger.info(f"Context length: {len(context)}")

            ans=self.generator.generate(query,context)

            return ans
        
        except Exception as e:
            logger.error("Error in RAG pipeline")
            raise CustomException(e,sys)