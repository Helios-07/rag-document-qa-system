from src.pipeline.rag_pipeline import RAGPipeine
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

def main():
    try:
        logger.info("Starting RAG Application")

        pipeline=RAGPipeine()

        while True:
            query=input("\nEnter your question (or type 'exit'): ")

            if query.lower()=="exit":
                print("Exiting...")
                break

            logger.info(f"User query: {query}")

            ans=pipeline.run(query)

            print("\n Answer:\n")
            print(ans)

    except Exception as e:
        logger.error("Application Failed")
        raise CustomException(e,sys)
    

if __name__=="__main__":
    main()