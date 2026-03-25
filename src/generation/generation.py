from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger=get_logger(__name__)

class Generator:
    def __init__(self, model_name, max_length=200):
        try:
            logger.info(f"Loading LLM: {model_name}")

            self.tokenizer=AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModelForSeq2SeqLM.from_pretrained(model_name)

            self.generator=pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer
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
You are a strict question-answering system.

Rules:
- Answer ONLY using the given context
- Give a COMPLETE and CORRECT answer
- Do NOT copy random sentences
- If definition is asked → give proper definition
- Keep answer clear and meaningful (3–5 lines)

Context:
{context}

Question:
{query}

Answer:
"""
            prompt = prompt[:1500]
            response=self.generator(
                prompt,
                max_length=self.max_length,
                do_sample=False
            )

            return response[0]['generated_text']
        
        except Exception as e:
            logger.error("Error generating response")
            raise CustomException(e,sys)