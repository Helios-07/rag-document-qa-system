from transformers import pipeline
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

class Generator:
    def __init__(self, model_name, max_length=200):
        try:
            logger.info(f"Loading LLM: {model_name}")

            self.generator=pipeline(
                "text2text-generation",
                model=model_name
            )

            self.max_length=max_length

            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error("Error loading LLM")
            raise CustomException(e,sys)
        

    def generate(self,query,context):
        try:
            logger.info("Generating answer")

            prompt = f"""
Act as an expert analyst. Examine the context below to answer the user's query.

### Context:
{context}

### Task:
1. First, identify the specific parts of the context relevant to the question.
2. Second, synthesize those parts into a clear, direct answer.
3. If the context contradicts itself or is missing info, point that out.

### Question:
{query}

### Detailed Answer:
"""
            response=self.generator(
                prompt,
                max_length=self.max_length,
                do_sample=False
            )

            return response[0]['generated_text']
        
        except Exception as e:
            logger.error("Error generating response")
            raise CustomException(e,sys)