from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import yaml
import os

logger=get_logger(__name__)

load_dotenv(override=True)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
class Generator:
    def __init__(self):
        try:
            config=load_config()

            gen_config=config['generation']

            self.model=gen_config['model_name']
            self.temp=gen_config['temperature']

            logger.info(f"Loading OpenAI model: {self.model}")

            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found in environment")

            self.client=OpenAI()

        except Exception as e:
            logger.error("Error initializing OpenAI client")
            raise CustomException(e,sys)
        
    
    def generate(self, query, context, history=""):
        try:
            logger.info("Generating answer using OpenAI")

            response=self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role':'system',
                        'content':"""
You are an expert assistant.

- Answer clearly and directly
- Do NOT mention the word "context"
- Start with a direct definition for "what is" questions
- If the answer is partially available, infer and complete it naturally
- Keep answers concise (2–4 lines)
"""
                    },

                    {
                        'role':'user',
                        "content": f"""
Answer the question using the information below.

Previous Conversation:
{history}

Context:
{context}

Question:
{query}

Answer:
"""
                    }
                ],
                temperature=self.temp,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error("Error generating answer")
            raise CustomException(e,sys)