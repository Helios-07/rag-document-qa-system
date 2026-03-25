from src.retrieval.retriever import Retriever
from src.generation.generation import Generator
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        try:
            logger.info("Initializing RAG Pipeline")

            self.retriever=Retriever()

            self.generator=Generator()

        except Exception as e:
            logger.error("Error initializing RAG pipeline")
            raise CustomException(e,sys)
        
    
    def run(self,query):
        try:
            logger.info(f"Running RAG pipeline for query: {query}")

            retrieved_chunks=self.retriever.retrieve(query, top_k=5)
            retrieved_chunks = [chunk for chunk in retrieved_chunks if len(chunk) > 50]
            #Debug
            print("\n--- Retrieved Chunks ---\n")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"[{i}] {chunk[:300]}\n")

            clean_chunks=[
                " ".join(chunk.replace("\n", " ").split())[:400]
                for chunk in retrieved_chunks
                if len(chunk.strip())>50
            ]
            logger.info(f"Retrieved {len(clean_chunks)} valid chunks")

            context = ""
            for chunk in clean_chunks[:3]:
                context+=chunk+"\n\n"
            
            context=" ".join(context.split()[:400])

            logger.info(f"Context length: {len(context)}")

            ans=self.generator.generate(query,context)

            return ans
        
        except Exception as e:
            logger.error("Error in RAG pipeline")
            raise CustomException(e,sys)